# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""GTGraph decorator
Interface functions to define the 'gtgraph' decorator to construct dataflow
graphs from functions that call gtscript stencils.
"""

import ast
import inspect
import warnings
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from uuid import uuid4

import astor
import cupy.cuda
from graphviz import Digraph

from gt4py import AccessKind, Boundary, FieldInfo
from gt4py.gtscript import stencil as gtstencil
from gt4py.stencil_object import StencilObject
from gt4py.storage import Storage


@dataclass
class InvokedStencil:
    stencil: StencilObject
    access_info: Tuple[Set[str], Set[str]]
    done_event: cupy.cuda.Event
    id: int
    region: Optional[Tuple[int, int, int, int]]  # x_lo, x_hi, y_lo, y_hi INCLUSIVE on both side


@dataclass
class LastAccessStencil:
    last_read_stencil_id: List[int]
    last_write_stencil_id: Optional[int]


class RuntimeGraph:
    def __init__(self, max_num_edges=10000):
        self.adjoint_nodes: List[Set[int]] = []
        self.connected_nodes: List[Set[int]] = []  # Transitive closure
        self._stencil_id_ticket = 0
        self._connected_num_edges = 0
        self._max_num_edges = max_num_edges

    def get_stencil_id(self):
        stencil_id = self._stencil_id_ticket
        self._stencil_id_ticket += 1
        self.adjoint_nodes.append(set())
        self.connected_nodes.append(set())
        return stencil_id

    def add_stencil(self, stencil_id: int, denpendencies: Iterable[int]):
        connected_set = set(denpendencies)
        connected_set.update(i for j in denpendencies for i in self.connected_nodes[j])
        self.connected_nodes[stencil_id] = connected_set
        self._connected_num_edges += len(connected_set)
        adjoint_set = set()
        for dep_i in sorted(denpendencies):
            adjoint_set.difference_update(self.connected_nodes[dep_i])
            adjoint_set.add(dep_i)
        self.adjoint_nodes[stencil_id] = adjoint_set

    def query_dependencies(self, stencil_id):
        return self.adjoint_nodes[stencil_id]  # stencil_id < self._stencil_id_ticket:

    def need_clean(self):
        return self._connected_num_edges > self._max_num_edges

    def clean_nonactive_stencils(self, active_stencils: Iterable[int]):
        active_stencils_set = set(active_stencils)
        for i in range(self._stencil_id_ticket):
            self._connected_num_edges -= len(self.connected_nodes[i])
            if i in active_stencils_set:
                self.connected_nodes[i].intersection_update(active_stencils_set)
                self._connected_num_edges += len(self.connected_nodes[i])
            else:
                self.connected_nodes[i] = set()


@dataclass
class AsyncStat:
    scheduled_stencils: int = 0
    finished_stencils: int = 0


class AsyncContext:
    def __init__(
        self,
        num_streams,
        max_invoked_stencils=50,
        blocking=False,
        concurrent=True,
        sleep_time=0.5,
        graph_record=False,
        name=None,
        profiling=False,
        region_analysis=False,
        validate_args=False,
    ):
        self.stream_pool: List[Optional[cupy.cuda.Stream]] = []
        self.runtime_graph: RuntimeGraph = RuntimeGraph()
        self.add_streams(num_streams)
        self.last_access_stencil: Dict[
            str, LastAccessStencil
        ] = dict()  # Dict[field_name, LastAccessStencil]
        self.invoked_stencils: Deque[InvokedStencil] = deque()
        self.max_invoked_stencils: int = max_invoked_stencils
        self.known_num_fields: int = 0
        self.blocking: bool = blocking  # synchronize all stencils, debug only
        self.concurrent: bool = concurrent  # only use root stream, debug only
        self.sleep_time: float = (
            sleep_time  # (second) longer sleep saves cpu cycle but lower resolution in timing
        )
        self.stat: AsyncStat = AsyncStat()
        self.region_analysis: bool = region_analysis
        self.validate_args: bool = validate_args
        self.signature_cache: Dict[str, inspect.Signature] = dict()
        self._graph_record: bool = False
        self._graph = None
        self.profiling: bool = profiling  # TODO: deal with concurrent stencil
        if name is None:
            name = uuid4().hex[:5]  # get a (almost) unique name
        self.name: str = name
        if graph_record:
            self.graph_record()

    def get_node_name(self, node_name: str, stencil_id: int):
        return f"{node_name}_cluster_{stencil_id}"

    def graph_record(self, filename: Optional[str] = None):
        self._graph_record = True
        if filename is None:
            filename = f"{self.name}_graph.gv"
        self._graph = Digraph(name=f"{self.name}_graph", filename=filename)

    def graph_add_stencil(
        self, stencil: StencilObject, bind_info: Dict[str, List[str]], stencil_id: int
    ):
        args_name = ",".join(bind_info["fields"])
        stencil_name = (
            f"{stencil.options['name']}_{stencil_id}({args_name})"  # {stencil.options['module']}_
        )
        row_ind, col_ind = self.get_kernel_dependencies(stencil)
        num_kernels = len(row_ind) - 1
        with self._graph.subgraph(name=f"cluster_{stencil_id}") as c:
            c.attr(style="filled", color="lightgrey")
            c.node_attr.update(style="filled", color="white")
            start_name = self.get_node_name("start", stencil_id)
            end_name = self.get_node_name("end", stencil_id)
            c.node(start_name, label="start", shape="box")
            c.node(end_name, label="end", shape="box")
            c.attr(label=stencil_name)
            for i in range(num_kernels):
                cols = col_ind[row_ind[i] : row_ind[i + 1]]
                name_i = self.get_node_name(f"kernel{i}", stencil_id)
                c.node(name_i, label=f"kernel {i}")
                if not cols:  # start only direct to those with out dependency
                    c.edge(start_name, name_i)
                if i not in col_ind:  # only those who does not have decendents points to end
                    c.edge(name_i, end_name)
                for j in cols:
                    name_j = self.get_node_name(f"kernel{j}", stencil_id)
                    c.edge(name_j, name_i)

    def graph_add_stencil_dependency(self, stencil_id_i: int, stencil_id_j: int):
        """
        i is dependent on j, j -> i
        """
        if stencil_id_i != stencil_id_j and stencil_id_i is not None and stencil_id_j is not None:
            node_name_i = self.get_node_name("start", stencil_id_i)
            node_name_j = self.get_node_name("end", stencil_id_j)
            self._graph.edge(node_name_j, node_name_i, style="bold", color="blue")

    def graph_stop_record(self):
        self._graph_record = False

    def graph_view(self, cleanup=True):
        if isinstance(self._graph, Digraph):
            self._graph.view(cleanup=cleanup)

    def graph_save(self, cleanup=True):
        if isinstance(self._graph, Digraph):
            self._graph.render(cleanup=cleanup, format="pdf")

    def add_streams(self, num_streams):
        self.stream_pool.extend(cupy.cuda.Stream(non_blocking=True) for _ in range(num_streams))

    def allocate_streams(self, num_streams):
        streams = []
        n = 0
        for stream in self.stream_pool:
            if n < num_streams:
                if stream.done:
                    streams.append(stream)
                    n += 1
            else:
                break
        if n < num_streams:
            self.add_streams(num_streams - n)
            streams.extend(self.stream_pool[n - num_streams :])
        return streams

    def free_finished_stencils(self):
        # https://stackoverflow.com/questions/8037455/how-to-modify-python-collections-by-filtering-in-place
        for _ in range(len(self.invoked_stencils)):
            stencil = self.invoked_stencils.popleft()
            if not stencil.done_event.done:
                self.invoked_stencils.append(stencil)
            else:
                self.stat.finished_stencils += 1

    def set_field_name(self, field: Storage, field_name: str):
        field._field_name = field_name

    def get_field_name(self, field: Storage):
        if not hasattr(field, "_field_name"):
            name = f"field_{self.known_num_fields}"
            self.known_num_fields += 1
            field._field_name = name
        return field._field_name

    def bind_arguments(self, signature: inspect.Signature, args, kwargs) -> List[Tuple[str, Any]]:
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

    def get_field_access_info(
        self,
        stencil_id: int,
        stencil: StencilObject,
        args: list,
        kwargs: dict,
        bind_info: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Set[str], Set[str]]:
        read_set = set()
        write_set = set()
        stencil_name = type(stencil).__name__
        if stencil_name in self.signature_cache:
            stencil_sig = self.signature_cache[stencil_name]
        else:
            stencil_sig = inspect.signature(stencil)
            self.signature_cache[stencil_name] = stencil_sig
        stencil_sig_bind = self.bind_arguments(stencil_sig, args, kwargs)
        # stencil_sig_bind = stencil_sig.bind(*args, **kwargs)
        for k, v in stencil_sig_bind:
            if k in stencil.field_info and stencil.field_info[k] is not None:
                field_name = self.get_field_name(v)
                access_kind = stencil.field_info[k].access
                if AccessKind.WRITE in access_kind:
                    write_set.add(field_name)
                elif AccessKind.READ in access_kind:
                    read_set.add(field_name)
                else:
                    raise ValueError("AccessKind not recognized!")
                if bind_info is not None:
                    bind_info["args"].append(k)
                    bind_info["fields"].append(field_name)
        return (read_set, write_set)

    def update_last_access_stencil(self, stencil_id: int, access_info: Tuple[Set[str], Set[str]]):
        reads, writes = access_info
        for field_name in reads.union(writes).difference(self.last_access_stencil.keys()):
            self.last_access_stencil[field_name] = LastAccessStencil([], None)
        for field_name in reads:
            self.last_access_stencil[field_name].last_read_stencil_id.append(stencil_id)
        for field_name in writes:
            self.last_access_stencil[field_name].last_write_stencil_id = stencil_id
            self.last_access_stencil[field_name].last_read_stencil_id = [stencil_id]

    def get_kernel_dependencies(
        self, stencil: StencilObject
    ) -> Tuple[List[int], List[int]]:  # (row_ind, col_ind)
        assert hasattr(stencil, "pyext_module")
        num_kernels = stencil.pyext_module.num_kernels()
        col_ind = []
        row_ind = [0] * (num_kernels + 1)
        if stencil.pyext_module.has_dependency_info():
            row_ind = stencil.pyext_module.dependency_row_ind()
            col_ind = stencil.pyext_module.dependency_col_ind()
            assert len(row_ind) == num_kernels + 1, "CSR format in dependency data is broken"
        return row_ind, col_ind

    def get_stencil_region(
        self,
        field_infos: Dict[str, FieldInfo],
        origin: Union[Tuple[int, int, int], Dict[str, Tuple[int, int, int]]],
        domain: Tuple[int, int, int],
    ):
        if isinstance(origin, dict):
            o_s = [o for o in origin.values() if len(o) > 1]
            if o_s:
                for o_i in o_s[1:]:
                    if (o_i[0] != o_s[0]) or (o_i[1] != o_s[1]):
                        # input 2D/3D fields have different origins
                        # ignore this case
                        # TODO: consider different regions
                        return None
            else:
                return None
        ext_x_lo = 0
        ext_x_hi = 0
        ext_y_lo = 0
        ext_y_hi = 0
        for field_info in field_infos.values():
            if field_info is not None:
                boundary = field_info.boundary
                ext_x_lo = max(ext_x_lo, boundary[0][0])
                ext_x_hi = max(ext_x_hi, boundary[0][1])
                ext_y_lo = max(ext_y_lo, boundary[1][0])
                ext_y_hi = max(ext_y_hi, boundary[1][1])
        x_lo = origin[0] - ext_x_lo
        x_hi = origin[0] + domain[0] - 1 + ext_x_hi
        y_lo = origin[1] - ext_y_lo
        y_hi = origin[1] + domain[1] - 1 + ext_y_hi
        return x_lo, x_hi, y_lo, y_hi

    def region_overlap(
        self,
        region1: Optional[Tuple[int, int, int, int]],
        region2: Optional[Tuple[int, int, int, int]],
    ) -> bool:
        if (region1 and region2) is None:
            return True
        if (
            region1[1] > region2[0]
            and region1[0] < region2[1]
            and region1[3] > region2[2]
            and region1[2] < region2[3]
        ):
            return True
        return False

    def get_dependencies(
        self,
        access_info: Tuple[Set[str], Set[str]],
        stencil_id: int,
        stencil_region: Optional[Tuple[int, int, int, int]],
    ) -> List[cupy.cuda.Event]:
        # R -> W, W -> W, W -> R
        dep_events = []
        reads, writes = access_info
        dep_set = set()
        no_dep_set = set()
        for stencil_i in self.invoked_stencils:
            if not stencil_i.done_event.done:
                if self.region_analysis and (
                    not self.region_overlap(stencil_region, stencil_i.region)
                ):
                    no_dep_set.add(stencil_i.id)
                    continue
                reads_i, writes_i = stencil_i.access_info
                dep_flag = False
                if writes.intersection(reads_i):  # R -> W
                    dep_flag = True
                elif writes.intersection(writes_i):  # W -> W
                    dep_flag = True
                elif reads.intersection(writes_i):  # W -> R
                    dep_flag = True
                if dep_flag:
                    dep_events.append(stencil_i.done_event)
                    dep_set.add(stencil_i.id)
        if self._graph_record:
            # In case some stencils have finished but still have influence on dependency
            for field in writes.intersection(self.last_access_stencil.keys()):
                # R -> W
                dep_set.update(self.last_access_stencil[field].last_read_stencil_id)
                # W -> W
                dep_set.add(self.last_access_stencil[field].last_write_stencil_id)
            for field in reads.intersection(self.last_access_stencil.keys()):
                # W -> R
                dep_set.add(self.last_access_stencil[field].last_write_stencil_id)
            dep_set.difference_update({stencil_id, None})
            dep_set.difference_update(no_dep_set)
            if self.runtime_graph.need_clean():
                active_stencil_set = set(stencil.id for stencil in self.invoked_stencils)
                active_stencil_set.update(
                    i for las in self.last_access_stencil.values() for i in las.last_read_stencil_id
                )
                active_stencil_set.update(
                    las.last_write_stencil_id for las in self.last_access_stencil.values()
                )
                active_stencil_set.discard(None)
                self.runtime_graph.clean_nonactive_stencils(active_stencil_set)
            self.runtime_graph.add_stencil(stencil_id, dep_set)
            dep_set = self.runtime_graph.query_dependencies(stencil_id)
            for j in dep_set:
                self.graph_add_stencil_dependency(stencil_id, j)
        return dep_events

    def add_invoked_stencil(
        self,
        stencil: StencilObject,
        access_info: Tuple[Set[str], Set[str]],
        done_event: cupy.cuda.Event,
        stencil_id: int,
        stencil_region: Optional[Tuple[int, int, int, int]],
    ):
        while len(self.invoked_stencils) >= self.max_invoked_stencils:
            sleep(self.sleep_time)  # wait for gpu computation
            self.free_finished_stencils()
        self.invoked_stencils.append(
            InvokedStencil(
                stencil=stencil,
                access_info=access_info,
                done_event=done_event,
                id=stencil_id,
                region=stencil_region,
            )
        )

    def wait(self):
        if self.concurrent:
            while len(self.invoked_stencils) > 0:
                sleep(self.sleep_time)
                self.free_finished_stencils()
        else:
            cupy.cuda.Stream.null.synchronize()
            self.stat.finished_stencils = self.stat.scheduled_stencils

    def wait_finish(self):
        self.wait()
        for i in range(len(self.stream_pool)):
            self.stream_pool[i] = None
        self.stream_pool = []

    def schedule(self, stencil: StencilObject, *args, **kwargs):
        if self.blocking:
            stencil(*args, **kwargs, validate_args=self.validate_args)
        elif stencil.backend == "gtc:cuda":
            if not self.concurrent:
                num_kernels = stencil.pyext_module.num_kernels()
                stencil(
                    *args,
                    async_launch=True,
                    streams=[0] * num_kernels,
                    validate_args=self.validate_args,
                    **kwargs,
                )
            elif not self.blocking:
                self.async_schedule(stencil, args, kwargs)
        else:
            warnings.warn(
                f"Backend: {stencil.backend} does not support async launching, use blocking behavior instead"
            )
            self.wait()
            stencil(*args, **kwargs, validate_args=self.validate_args)
        self.stat.scheduled_stencils += 1

    def async_schedule(self, stencil: StencilObject, args, kwargs):
        """
        Step 0: remove finished calls & free streams
        Step 1: mark fields if first meet
        Step 2: analyse dependency
        Step 3: allocate streams
        Step 4: insert start & wait events in streams
        Step 5: invoke stencil
        Step 6: insert stop & wait events
        """
        # remove finished calls
        self.free_finished_stencils()

        # check stencil obj is generated by the right backend
        has_kernel_dependency_info = False

        assert hasattr(
            stencil, "pyext_module"
        ), f"The stencil object {stencil.__module__}.{stencil.__name__} is not generated by GTC:CUDA backend"
        num_kernels = stencil.pyext_module.num_kernels()
        has_kernel_dependency_info = stencil.pyext_module.has_dependency_info()

        # resolve dependency
        stencil_id = self.runtime_graph.get_stencil_id()
        bind_info = {"args": [], "fields": []} if self._graph_record else None
        access_info = self.get_field_access_info(
            stencil_id, stencil, args, kwargs, bind_info=bind_info
        )
        stencil_region = None
        if "origin" in kwargs and "domain" in kwargs and self.region_analysis:
            stencil_region = self.get_stencil_region(
                stencil.field_info, kwargs["origin"], kwargs["domain"]
            )

        if self._graph_record:
            self.graph_add_stencil(stencil, bind_info, stencil_id)

        dep_events = self.get_dependencies(access_info, stencil_id, stencil_region)
        # dep_events = [stencil.done_event for stencil in self.invoked_stencils if not stencil.done_event.done]
        self.update_last_access_stencil(stencil_id, access_info)

        # count how many streams needed
        num_streams = (
            num_kernels if has_kernel_dependency_info else 1
        )  # TODO: reduce unnecessary streams
        stream_pool = self.allocate_streams(num_streams)

        # insert events for waiting dependencies
        for stream in stream_pool:
            for dep_event in dep_events:
                stream.wait_event(dep_event)

        # Launch stencil
        if num_streams == 1:
            streams = stream_pool * num_kernels
        else:
            streams = stream_pool
        stream_ptrs = [stream.ptr for stream in streams]
        stencil(
            *args,
            async_launch=True,
            streams=stream_ptrs,
            validate_args=self.validate_args,
            **kwargs,
        )

        # insert events to record when the stencil finishes
        done_events = [
            cupy.cuda.Event(block=False, disable_timing=True) for _ in range(num_streams)
        ]
        for i in range(1, num_streams):
            done_events[i].record(stream_pool[i])
            stream_pool[0].wait_event(done_events[i])
        done_events[0].record(stream_pool[0])

        # update async_ctx
        self.add_invoked_stencil(stencil, access_info, done_events[0], stencil_id, stencil_region)


def async_stencil(async_context: AsyncContext, *, backend, **gtstencil_kwargs):
    def decorator(func):
        stencil = gtstencil(backend, func, **gtstencil_kwargs)

        def wrapper(*args, **kwargs):
            async_context.schedule(stencil, *args, **kwargs)

        return wrapper

    return decorator


class InsertAsync(ast.NodeTransformer):
    @classmethod
    def apply(cls, definition, ctx, num_streams_init=20):
        maker = cls(definition, ctx, num_streams_init)
        maker.ast_root = maker.visit(maker.ast_root)
        maker.ast_root = maker.insert_init(maker.ast_root)
        return astor.to_source(maker.ast_root, add_line_information=True)

    def __init__(self, definition, ctx, num_streams_init):
        # check AsyncContext and async_invoke is in ctx as well
        # if ("AsyncContext" not in ctx) and ("async_invoke" not in ctx):
        #    raise ValueError("Please import `AsyncContext` and `async_invoke` first")
        self.ast_root = astor.code_to_ast(definition)
        self.stencil_ctx = {
            k: ctx[k]
            for k in ctx
            if isinstance(ctx[k], StencilObject) and hasattr(ctx[k], "pyext_module")
        }
        self.num_streams_init = num_streams_init

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            keywords = [i.arg for i in node.keywords]
            if (func_name in self.stencil_ctx) and ("async_launch" not in keywords):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="async_context", ctx=ast.Load()),
                            attr="schedule",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id="async_context", ctx=ast.Load()), node.func] + node.args,
                        keywords=node.keywords,
                    ),
                    node,
                )
        return node

    def insert_init(self, node: ast.FunctionDef):
        # num_streams_guess = min(20, sum(eval(stencil).pyext_module.num_kernels() for stencil in self.stencil_ctx))
        import_node = ast.ImportFrom(
            module="gt4py.gtgraph",
            names=[ast.alias(name="AsyncContext", asname=None)],
            level=0,
            lineno=node.body[0].lineno,
        )
        call_node = ast.Call(
            func=ast.Name(id="AsyncContext", ctx=ast.Load()),
            args=[ast.Constant(value=self.num_streams_init, kind=None)],
            keywords=[],
        )
        start_node = ast.Assign(
            targets=[ast.Name(id="async_context", ctx=ast.Store())],
            value=call_node,
            lineno=node.body[0].lineno,
        )
        end_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="async_context", ctx=ast.Load()),
                    attr="wait_finish",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            lineno=node.body[-1].lineno,
        )
        new_node = ast.copy_location(
            ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=[import_node, start_node] + node.body + [end_node],
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
            ),
            node,
        )
        return new_node
