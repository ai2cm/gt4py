# distutils: language = c++
# cython: language_level=3
# cython: profile=True, linetrace=True, binding=True
from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libcpp.unordered_set cimport unordered_set as cpp_set

from libc.stdint cimport intptr_t

from gt4py.stencil_object import StencilObject
from gt4py.storage import Storage
from gt4py import AccessKind, FieldInfo
import inspect
from time import sleep
import warnings
from typing import Iterable, Collection, Optional, Tuple

#from line_profiler_pycharm import profile

from .cuda_runtime cimport *

"""
Cython Implementation of AsyncContext

Annotate with `cythonize -a gtgraph_fast.pyx`
"""

ctypedef struct access_info_t:
    cpp_set[int] reads
    cpp_set[int] writes
    bint blocking_access

ctypedef struct InvokedStencil:
    #cdef StencilObject stencil
    access_info_t access_info
    intptr_t done_event
    int id

def set_field_name(field: Storage, field_name: str):
    field._field_name = field_name

cdef bint set_intersection(cpp_set[int] set1, cpp_set[int] set2):
    for e in set1:
        if set2.find(e) != set2.end():
            return True
    return False

cdef class AsyncContextFast:

    cdef vector[intptr_t] stream_pool

    cdef deque[InvokedStencil] invoked_stencils

    cdef int max_invoked_stencils

    cdef int known_num_fields, num_scheduled_stencils

    cdef bint blocking, concurrent, validate_args

    cdef dict signature_cache

    cdef double sleep_time

    def __init__(self, int num_streams, int max_invoked_stencils = 50, bint blocking = False, bint concurrent = True, double sleep_time = 0.5, bint validate_args=False, **kwargs):
        self.add_streams(num_streams)
        self.max_invoked_stencils = max_invoked_stencils
        self.known_num_fields = 0
        self.num_scheduled_stencils = 0
        self.blocking = blocking
        self.concurrent = concurrent
        self.signature_cache = dict()
        self.sleep_time = sleep_time

    cdef add_streams(self, int num_streams):
        cdef int i
        self.stream_pool.reserve(self.stream_pool.size() + num_streams)
        for i in range(num_streams):
            stream_ptr = streamCreateWithFlags(0x1) # 1 for nonblocking
            self.stream_pool.push_back(stream_ptr)

    cdef vector[intptr_t] allocate_streams(self, int num_streams):
        cdef vector[intptr_t] streams
        streams.reserve(num_streams)
        n = 0
        for stream in self.stream_pool:
            if n < num_streams:
                if streamQuery(stream) == 0:
                    streams.push_back(stream)
                    n += 1
            else:
                break
        if n < num_streams:
            self.add_streams(num_streams - n)
            for i in range(n, num_streams):
                streams.push_back(self.stream_pool[i])
        return streams

    cdef free_finished_stencils(self):
        cdef InvokedStencil stencil
        for _ in range(self.invoked_stencils.size()):
            stencil = self.invoked_stencils[0]
            self.invoked_stencils.pop_front()
            if eventQuery(stencil.done_event) != 0:
                self.invoked_stencils.push_back(stencil)
            else:
                eventDestroy(stencil.done_event)

    cdef int get_field_id(self, field: Storage):
        cdef int id
        if not hasattr(field, '_field_id'):
            id = self.known_num_fields
            self.known_num_fields += 1
            field._field_id = id
        else:
            id = field._field_id
        return id

    cdef list bind_arguments(self, signature: inspect.Signature, args: Collection, kwargs: dict):
        n = len(args)
        i = 0
        sig_bind = []
        for param in signature.parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD:
                if i < n:
                    sig_bind.append((param.name, args[i]))
                elif param.name in kwargs and param.default is param.empty:
                    sig_bind.append((param.name, kwargs[param.name]))
                i += 1
            elif i < n:
                raise ValueError("Excess positional arguments have been sepcified!")
            elif param.kind == param.KEYWORD_ONLY and param.name in kwargs:
                sig_bind.append((param.name, kwargs[param.name]))
        return sig_bind

    cdef get_field_access_info(self, int stencil_id, stencil: StencilObject, args: Collection, kwargs: dict, access_info_t& access_info):
        stencil_name = type(stencil).__name__
        if stencil_name in self.signature_cache:
            stencil_sig = self.signature_cache[stencil_name]
        else:
            stencil_sig = inspect.signature(stencil)
            self.signature_cache[stencil_name] = stencil_sig
        stencil_sig_bind = self.bind_arguments(stencil_sig, args, kwargs)
        #stencil_sig_bind = stencil_sig.bind(*args, **kwargs)
        cdef int field_id
        for k, v in stencil_sig_bind:
            if k in stencil.field_info and stencil.field_info[k] is not None:
                field_id = self.get_field_id(v)
                access_kind = stencil.field_info[k].access
                if AccessKind.WRITE in access_kind:
                    access_info.writes.insert(field_id)
                elif AccessKind.READ in access_kind:
                    access_info.reads.insert(field_id)
                else:
                    raise ValueError("AccessKind not recognized!")

    cdef get_field_access_info_fast(self, int stencil_id, stencil: StencilObject, args: Collection, arg_names: Collection, access_info_t& access_info):
        cdef int field_id, i
        for i in range(len(args)):
            k = arg_names[i]
            v = args[i]
            if k in stencil.field_info and stencil.field_info[k] is not None:
                field_id = self.get_field_id(v)
                access_kind = stencil.field_info[k].access
                if AccessKind.WRITE in access_kind:
                    access_info.writes.insert(field_id)
                    v.host_to_device()
                    v._set_device_modified()
                elif AccessKind.READ in access_kind:
                    access_info.reads.insert(field_id)
                else:
                    raise ValueError("AccessKind not recognized!")

    cdef get_dependencies(self, access_info_t access_info, int stencil_id, vector[intptr_t]& dep_events):
        # R -> W, W -> W, W -> R
        #cdef vector[intptr_t] dep_events
        cdef bint dep_flag
        for stencil_i in self.invoked_stencils:
            #if eventQuery(stencil_i.done_event) != 0:
            if stencil_i.access_info.blocking_access:
                dep_flag = True
            elif set_intersection(access_info.writes, stencil_i.access_info.reads): # R -> W
                dep_flag = True
            elif set_intersection(access_info.writes, stencil_i.access_info.writes): # W -> W
                dep_flag = True
            elif set_intersection(access_info.reads, stencil_i.access_info.writes): # W -> R
                dep_flag = True
            if dep_flag:
                dep_events.push_back(stencil_i.done_event)
        #return dep_events

    cdef add_invoked_stencil(self, access_info_t access_info, intptr_t done_event, int stencil_id):
        while self.invoked_stencils.size() >= <unsigned int> self.max_invoked_stencils:
            sleep(self.sleep_time)
            self.free_finished_stencils()
        self.invoked_stencils.push_back(InvokedStencil(access_info=access_info, done_event= done_event, id=stencil_id))

    cpdef wait(self):
        if self.concurrent:
            while self.invoked_stencils.size() > 0:
                sleep(self.sleep_time)
                self.free_finished_stencils()
        else:
            streamSynchronize(0)

    cpdef wait_finish(self):
        self.wait()
        if self.stream_pool.size() > 0:
            for stream_ptr in self.stream_pool:
                streamDestroy(stream_ptr)
            self.stream_pool.clear()

    def __dealloc__(self):
        self.wait_finish()

    def schedule(self, stencil: StencilObject, *args,
                 fast_schedule_info: Optional[Tuple[Collection, dict, tuple]] = None,
                 **kwargs):
        if self.blocking:
            stencil(*args, **kwargs, validate_args=self.validate_args)
        elif stencil.backend == "gtc:cuda":
            if not self.concurrent:
                num_kernels = stencil.pyext_module.num_kernels()
                stencil(*args, async_launch=True, streams=[0]*num_kernels, validate_args=self.validate_args, **kwargs)
            elif not self.blocking:
                if fast_schedule_info is not None:
                    arg_names, field_origins, domain = fast_schedule_info
                    kwargs.pop("domain", None)
                    kwargs.pop("origin", None)
                    self.async_schedule(stencil, args, kwargs, True, True, arg_names, field_origins, domain)
                else:
                    self.async_schedule(stencil, args, kwargs)
        else:
            warnings.warn(f"Backend: {stencil.backend} does not support async launching, use blocking behavior instead")
            self.wait()
            stencil(*args, **kwargs, validate_args=self.validate_args)

    #@profile
    def async_schedule(self, stencil: StencilObject, args: Collection, kwargs: dict,
                       fast_analysis: bool = False, call_run: bool = False,
                       arg_names: Optional[Collection] = None,
                       field_origins: Optional[dict] = None,
                       domain: Optional[Tuple[int, int, int]] = None):
        self.free_finished_stencils()

        cdef int num_kernels = stencil.pyext_module.num_kernels()
        cdef int stencil_id = self.num_scheduled_stencils
        self.num_scheduled_stencils += 1
        cdef cpp_set[int] read_set
        cdef cpp_set[int] write_set
        cdef int num_args_approx = len(args) + len(kwargs)
        read_set.reserve(num_args_approx)
        write_set.reserve(num_args_approx)
        cdef access_info_t access_info = access_info_t(read_set, write_set, True)
        cdef vector[intptr_t] dep_events

        cdef int num_streams = num_kernels
        cdef vector[intptr_t] streams = self.allocate_streams(num_streams)

        if self.invoked_stencils.size() > 0:
            #TODO: do analysis for *expensive* stencil
            if fast_analysis:
                self.get_field_access_info_fast(stencil_id, stencil, args, arg_names, access_info)
            else:
                self.get_field_access_info(stencil_id, stencil, args, kwargs, access_info)
            access_info.blocking_access = False
            self.get_dependencies(access_info, stencil_id, dep_events)

            for stream in streams:
                for dep_event in dep_events:
                    streamWaitEvent(stream, dep_event)

        if call_run:
            kwargs["exec_info"] = None
            stencil.run(_domain_=domain, _origin_=field_origins, async_launch=True, streams=list(streams), **kwargs, **zip(arg_names, args))
        else:
            stencil(*args, async_launch=True, streams=list(streams), validate_args=self.validate_args, **kwargs)

        cdef intptr_t done_event_i
        for i in range(1, num_streams):
            done_event_i = eventCreateWithFlags(0x02) # Disable Timing
            eventRecord(done_event_i, streams[i]) # TODO: Recycle events using cyclic buffer
            streamWaitEvent(streams[0], done_event_i)
        cdef intptr_t done_event = eventCreateWithFlags(0x02)
        eventRecord(done_event, streams[0])

        self.add_invoked_stencil(access_info, done_event, stencil_id)
