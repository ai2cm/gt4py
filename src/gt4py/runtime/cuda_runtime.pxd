# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport intptr_t
from libc.stdio cimport printf
import cython

cdef extern from "cuda_runtime.h" nogil:
    ctypedef int Error 'cudaError_t'
    ctypedef void * Stream 'cudaStream_t'
    ctypedef void * Event 'cudaEvent_t'
    Error cudaStreamCreateWithFlags(Stream*, unsigned int)
    Error cudaStreamDestroy(Stream)
    Error cudaStreamSynchronize(Stream)
    Error cudaStreamQuery(Stream)
    Error cudaStreamWaitEvent(Stream, Event, unsigned int)
    Error cudaEventCreateWithFlags(Event*, unsigned int)
    Error cudaEventRecord(Event, Stream)
    Error cudaEventQuery(Event)
    Error cudaEventDestroy(Event)
    const char* cudaGetErrorName(Error error)
    const char* cudaGetErrorString(Error error)
    Error cudaGetLastError()

@cython.profile(False)
cdef inline check_status(Error status):
    if <int>status != 0:
        # to reset error status
        cudaGetLastError()
        printf("CUDA ERROR %s: %s\n", cudaGetErrorName(<Error>status), cudaGetErrorString(<Error>status))
        raise ValueError("CUDA ERROR")

cdef inline intptr_t streamCreateWithFlags(unsigned int flags) except? 0:
    cdef intptr_t stream
    status = cudaStreamCreateWithFlags(<Stream*>&stream, flags)
    check_status(status)
    return stream

cdef inline streamDestroy(intptr_t stream):
    status = cudaStreamDestroy(<Stream>stream)
    check_status(status)

cdef inline int streamQuery(intptr_t stream):
    return <int>cudaStreamQuery(<Stream>stream)

cdef inline streamSynchronize(intptr_t stream):
    with nogil:
        status = cudaStreamSynchronize(<Stream>stream)
    check_status(status)

cdef inline streamWaitEvent(intptr_t stream, intptr_t event):
    status = cudaStreamWaitEvent(<Stream>stream, <Event>event, 0)
    check_status(status)

cdef inline intptr_t eventCreateWithFlags(unsigned int flags) except? 0:
    cdef intptr_t event
    status = cudaEventCreateWithFlags(<Event*>&event, flags)
    check_status(status)
    return event

cdef inline eventRecord(intptr_t event, intptr_t stream = 0):
    status = cudaEventRecord(<Event>event, <Stream>stream)
    check_status(status)

cdef inline int eventQuery(intptr_t event):
    return <int>cudaEventQuery(<Event>event)

cdef inline eventDestroy(intptr_t event):
    status = cudaEventDestroy(<Event>event)
    check_status(status)