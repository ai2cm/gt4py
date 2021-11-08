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

"""Implementation of GTScript: an embedded DSL in Python for stencil computations.

Interface functions to define and compile GTScript definitions and empty symbol
definitions for the keywords of the DSL.
"""

import collections
import copy
import inspect
import numbers
import os
import re
import types
from typing import Callable, Dict, Type, Union

import dace
import numpy as np

import gt4py.backend
from gt4py import definitions as gt_definitions
from gt4py.lazy_stencil import LazyStencil
from gt4py.stencil_builder import StencilBuilder
from gt4py.utils import shash


# GTScript builtins
MATH_BUILTINS = {
    "abs",
    "min",
    "max",
    "mod",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sqrt",
    "exp",
    "log",
    "isfinite",
    "isinf",
    "isnan",
    "floor",
    "ceil",
    "trunc",
}

builtins = {
    "I",
    "J",
    "K",
    "IJ",
    "IK",
    "JK",
    "IJK",
    "FORWARD",
    "BACKWARD",
    "PARALLEL",
    "Field",
    "Sequence",
    "externals",
    "computation",
    "interval",
    "horizontal",
    "region",
    "__gtscript__",
    "__externals__",
    "__INLINED",
    "compile_assert",
    "index",
    "range",
    *MATH_BUILTINS,
}

IGNORE_WHEN_INLINING = {*MATH_BUILTINS, "compile_assert", "index", "range", "K"}

__all__ = list(builtins) + ["function", "stencil", "lazy_stencil"]

__externals__ = "Placeholder"
__gtscript__ = "Placeholder"

_VALID_DATA_TYPES = (
    bool,
    np.bool_,
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
)


def _set_arg_dtypes(definition: Callable[..., None], dtypes: Dict[Type, Type]):
    assert isinstance(definition, types.FunctionType)
    annotations = getattr(definition, "__annotations__", {})
    original_annotations = {**annotations}
    for arg, value in annotations.items():
        if isinstance(value, _FieldDescriptor) and isinstance(value.dtype, str):
            if value.dtype in dtypes:
                annotations[arg] = _FieldDescriptor(
                    dtypes[value.dtype], value.axes, value.data_dims
                )
            else:
                raise ValueError(f"Missing '{value.dtype}' dtype definition for arg '{arg}'")
        elif isinstance(value, str):
            if value in dtypes:
                annotations[arg] = dtypes[value]
            else:
                raise ValueError(f"Missing '{value}' dtype definition for arg '{arg}'")

    return original_annotations


def function(func):
    """GTScript function."""

    from gt4py.frontend import gtscript_frontend as gt_frontend

    gt_frontend.GTScriptParser.annotate_definition(func)
    return func


# Interface functions
def stencil(
    backend,
    definition=None,
    *,
    build_info=None,
    dtypes=None,
    externals=None,
    format_source=True,
    name=None,
    rebuild=False,
    **kwargs,
):
    """Generate an implementation of the stencil definition with the specified backend.

    It can be used as a parametrized function decorator or as a regular function.

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        build_info : `dict`, optional
            Dictionary used to store information about the stencil generation.
            (`None` by default). Possible key-value pairs include:
            - 'def_ir': (StencilDefinition) Definition IR object
            - 'iir': (StencilImplementation) Implementation IR object
            - 'symbol_info': (Dict[str, SymbolInfo]) Dictionary of SymbolInfo objects
            - 'parse_time': (float) Frontend run time, e.g., parsing GTScript in seconds
            - 'module_time': (float) Python module generation time in seconds
            - 'codegen_time'" (float) Backend-specific code generation time in seconds
            - 'build_time': (float) Compilation time, i.e., for non-Python backends in seconds
            - 'load_time': (float) Module load time for cached stencils in seconds

        dtypes: `dict`[`str`, dtype_definition], optional
            Specify dtypes for string keys in the argument annotations.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        format_source : `bool`, optional
            Format generated sources when possible (`True` by default).

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`gridtools.StencilObject`
            Properly initialized instance of a dynamically-generated
            subclass of :class:`gt4py.StencilObject`.

    Raises
    -------
        ValueError
            If inconsistent arguments are specified.

    Examples
    --------
        TODO

    """

    from gt4py import loader as gt_loader

    if build_info is not None and not isinstance(build_info, dict):
        raise ValueError(f"Invalid 'build_info' dictionary ('{build_info}')")
    if dtypes is not None and not isinstance(dtypes, dict):
        raise ValueError(f"Invalid 'dtypes' dictionary ('{dtypes}')")
    if externals is not None and not isinstance(externals, dict):
        raise ValueError(f"Invalid 'externals' dictionary ('{externals}')")
    if not isinstance(format_source, bool):
        raise ValueError(f"Invalid 'format_source' bool value ('{name}')")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"Invalid 'name' string ('{name}')")
    if not isinstance(rebuild, bool):
        raise ValueError(f"Invalid 'rebuild' bool value ('{rebuild}')")

    module = None
    if name:
        name_components = name.split(".")
        name = name_components[-1]
        module = ".".join(name_components[:-1])

    name = name or ""
    module = (
        module or inspect.currentframe().f_back.f_globals["__name__"]
    )  # definition_func.__globals__["__name__"] ??,

    # Move hidden "_option" keys to _impl_opts
    _impl_opts = {}
    for key, value in kwargs.items():
        if key.startswith("_"):
            _impl_opts[key] = value
    for key in _impl_opts:
        kwargs.pop(key)

    # Setup build_info timings
    if build_info is not None:
        time_keys = ("parse_time", "module_time", "codegen_time", "build_time", "load_time")
        build_info.update({time_key: 0.0 for time_key in time_keys})

    build_options = gt_definitions.BuildOptions(
        name=name,
        module=module,
        format_source=format_source,
        rebuild=rebuild,
        backend_opts=kwargs,
        build_info=build_info,
        impl_opts=_impl_opts,
    )

    def _decorator(definition_func):
        if not isinstance(definition_func, types.FunctionType):
            if hasattr(definition_func, "definition_func"):  # StencilObject
                definition_func = definition_func.definition_func
            elif callable(definition_func):  # General callable
                definition_func = definition_func.__call__

        original_annotations = _set_arg_dtypes(definition_func, dtypes or {})
        out = gt_loader.gtscript_loader(
            definition_func,
            backend=backend,
            build_options=build_options,
            externals=externals or {},
        )
        setattr(definition_func, "__annotations__", original_annotations)
        return out

    if definition is None:
        return _decorator
    else:
        return _decorator(definition)


def lazy_stencil(
    backend=None,
    definition=None,
    *,
    build_info=None,
    dtypes=None,
    externals=None,
    name=None,
    rebuild=False,
    eager=False,
    check_syntax=True,
    **kwargs,
):
    """
    Create a stencil object with deferred building and optional up-front syntax checking.

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        build_info : `dict`, optional
            Dictionary used to store information about the stencil generation.
            (`None` by default).

        dtypes: `dict`[`str`, dtype_definition], optional
            Specify dtypes for string keys in the argument annotations.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        name : `str`, optional
            The fully qualified name of the generated :class:`StencilObject`.
            If `None`, it will be set to the qualified name of the definition function.
            (`None` by default).

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        eager : `bool`, optional
            If true do not defer stencil building and instead return the fully built raw implementation object.

        check_syntax: `bool`, default=True, optional
            If true, build and cache the IR build stage already, which checks stencil definition syntax.

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`gridtools.build.LazyStencil`
            Wrapper around an instance of the dynamically-generated subclass of :class:`gt4py.StencilObject`.
            Defers the generation step until the last moment and allows syntax checking independently.
            Also gives access to a more fine grained generate / build process.
    """
    from gt4py import frontend

    def _decorator(func):
        _set_arg_dtypes(func, dtypes or {})
        options = gt_definitions.BuildOptions(
            **{
                **StencilBuilder.default_options_dict(func),
                **StencilBuilder.name_to_options_args(name),
                "rebuild": rebuild,
                "build_info": build_info,
                **StencilBuilder.nest_impl_options(kwargs),
            }
        )
        stencil = LazyStencil(
            StencilBuilder(func, backend=backend, options=options).with_externals(externals or {})
        )
        if eager:
            stencil = stencil.implementation
        elif check_syntax:
            stencil.check_syntax()
        return stencil

    if definition is None:
        return _decorator
    return _decorator(definition)


def as_sdfg(*args, **kwargs) -> dace.SDFG:
    def _decorator(definition_func):

        from gt4py.backend.gtc_backend.dace.backend import expand_and_wrap_sdfg, to_device
        from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
        from gt4py.definitions import BuildOptions
        from gt4py.frontend.gtscript_frontend import GTScriptFrontend
        from gtc import gtir_to_oir
        from gtc.dace.oir_to_dace import OirSDFGBuilder
        from gtc.passes.gtir_pipeline import GtirPipeline
        from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
        from gtc.passes.oir_optimizations.inlining import MaskInlining
        from gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
        from gtc.passes.oir_pipeline import DefaultPipeline

        default_pipeline = DefaultPipeline(
            skip=[
                MaskStmtMerging,
                MaskInlining,
                FillFlushToLocalKCaches,
            ]
        )
        oir_pipeline = kwargs.pop("oir_pipeline", default_pipeline)

        definition_ir = GTScriptFrontend.generate(
            definition_func,
            externals=kwargs,
            options=BuildOptions(
                name=definition_func.__name__,
                module=inspect.currentframe().f_back.f_globals["__name__"],
            ),
        )

        gt_ir = DefIRToGTIR.apply(definition_ir)
        gt_ir = GtirPipeline(gt_ir).full()
        base_oir = gtir_to_oir.GTIRToOIR().visit(gt_ir)
        oir = oir_pipeline.run(base_oir)

        sdfg: dace.SDFG = OirSDFGBuilder().visit(oir)
        backend = gt4py.backend.from_name(kwargs.get("backend", "gtc:dace"))
        to_device(sdfg, device=backend.storage_info["device"])
        sdfg = expand_and_wrap_sdfg(gt_ir, sdfg, layout_map=backend.storage_info["layout_map"])

        return sdfg

    if not kwargs and len(args) == 1:
        return _decorator(args[0])
    else:
        return _decorator


class SDFGWrapper:

    loaded_compiled_sdfgs: Dict[str, dace.SDFG] = dict()

    def __init__(
        self,
        definition,
        domain,
        origin,
        *,
        dtypes=None,
        externals=None,
        format_source=True,
        name=None,
        rebuild=False,
        **kwargs,
    ):

        self.func = definition

        self.domain = domain
        self.origin = origin
        self.backend = kwargs.get("backend", "gtc:dace")
        if "backend" in kwargs:
            del kwargs["backend"]
        self.device = gt4py.backend.from_name(self.backend).storage_info["device"]
        self.stencil_kwargs = {
            **kwargs,
            **dict(
                dtypes=dtypes,
                format_source=format_source,
                name=name,
                rebuild=rebuild,
                externals=externals,
            ),
        }
        self.stencil_object = None
        self.filename = None
        self._sdfg = None

    def __sdfg__(self, *args, **kwargs):

        if self.stencil_object is None:
            stencil_function = kwargs.pop("stencil_function", stencil)
            self.stencil_object = stencil_function(
                definition=self.func, backend="gtc:numpy", **self.stencil_kwargs
            )

            basename = os.path.splitext(self.stencil_object._file_name)[0]
            self.filename = (
                basename + "_wrapper_" + str(shash(self.device, self.origin, self.domain)) + ".sdfg"
            )

        # check if same sdfg already cached in memory
        if self._sdfg is not None:
            return copy.deepcopy(self._sdfg)
        elif self.filename in SDFGWrapper.loaded_compiled_sdfgs:
            self._sdfg = SDFGWrapper.loaded_compiled_sdfgs[self.filename]
            return copy.deepcopy(self._sdfg)

        # check if same sdfg already cached on disk
        if kwargs.get("use_disk_sdfg", False):
            try:
                self._sdfg = dace.SDFG.from_file(self.filename)
                print("reused (__sdfg__):", self.filename)
                SDFGWrapper.loaded_compiled_sdfgs[self.filename] = self._sdfg
                return copy.deepcopy(self._sdfg)
            except FileNotFoundError:
                pass
            except Exception:
                raise

        # otherwise, wrap and save sdfg from scratch
        inner_sdfg = as_sdfg(backend=self.backend, **(self.stencil_kwargs.get("externals", {})))(
            self.func
        )
        self._sdfg = dace.SDFG("SDFGWrapper_" + inner_sdfg.name)
        state = self._sdfg.add_state("SDFGWrapper_" + inner_sdfg.name + "_state")

        inputs = set()
        outputs = set()
        for inner_state in inner_sdfg.nodes():
            for node in inner_state.nodes():
                if (
                    not isinstance(node, dace.nodes.AccessNode)
                    or inner_sdfg.arrays[node.data].transient
                ):
                    continue
                if node.access != dace.dtypes.AccessType.WriteOnly:
                    inputs.add(node.data)
                if node.access != dace.dtypes.AccessType.ReadOnly:
                    outputs.add(node.data)

        nsdfg = state.add_nested_sdfg(inner_sdfg, None, inputs, outputs)
        for name, array in inner_sdfg.arrays.items():
            if isinstance(array, dace.data.Array) and not array.transient:
                axes = self.stencil_object.field_info[name].axes

                shape = [f"__{name}_{axis}_size" for axis in axes] + [
                    str(d) for d in self.stencil_object.field_info[name].data_dims
                ]

                self._sdfg.add_array(
                    name,
                    dtype=array.dtype,
                    strides=array.strides,
                    shape=shape,
                    storage=dace.StorageType.GPU_Global
                    if self.device == "gpu"
                    else dace.StorageType.Default,
                )
                if isinstance(self.origin, tuple):
                    origin = [o for a, o in zip("IJK", self.origin) if a in axes]
                else:
                    origin = self.origin.get(name, self.origin.get("_all_", None))
                    if len(origin) == 3:
                        origin = [o for a, o in zip("IJK", origin) if a in axes]

                subset_strs = [
                    f"{o - e}:{o - e + s}"
                    for o, e, s in zip(
                        origin,
                        self.stencil_object.field_info[name].boundary.lower_indices,
                        inner_sdfg.arrays[name].shape,
                    )
                ]
                subset_strs += [f"0:{d}" for d in self.stencil_object.field_info[name].data_dims]

                if name in inputs:
                    state.add_edge(
                        state.add_read(name),
                        None,
                        nsdfg,
                        name,
                        dace.Memlet.simple(name, ",".join(subset_strs)),
                    )
                if name in outputs:
                    state.add_edge(
                        nsdfg,
                        name,
                        state.add_write(name),
                        None,
                        dace.Memlet.simple(name, ",".join(subset_strs)),
                    )

        for symbol in nsdfg.sdfg.free_symbols:
            if symbol not in self._sdfg.symbols:
                self._sdfg.add_symbol(symbol, nsdfg.sdfg.symbols[symbol])

        if any(d == 0 for d in self.domain):
            states = self._sdfg.states()
            assert len(states) == 1
            for node in states[0].nodes():
                state.remove_node(node)
        ival, jval, kval = self.domain[0], self.domain[1], self.domain[2]
        for sdfg in self._sdfg.all_sdfgs_recursive():
            if sdfg.parent_nsdfg_node is not None:
                symmap = sdfg.parent_nsdfg_node.symbol_mapping

                if "__I" in symmap:
                    ival = symmap["__I"]
                    del symmap["__I"]
                if "__J" in symmap:
                    jval = symmap["__J"]
                    del symmap["__J"]
                if "__K" in symmap:
                    kval = symmap["__K"]
                    del symmap["__K"]

            sdfg.replace("__I", ival)
            if "__I" in sdfg.symbols:
                sdfg.remove_symbol("__I")
            sdfg.replace("__J", jval)
            if "__J" in sdfg.symbols:
                sdfg.remove_symbol("__J")
            sdfg.replace("__K", kval)
            if "__K" in sdfg.symbols:
                sdfg.remove_symbol("__K")

            for val in ival, jval, kval:
                sym = dace.symbolic.pystr_to_symbolic(val)
                for fsym in sym.free_symbols:
                    if sdfg.parent_nsdfg_node is not None:
                        sdfg.parent_nsdfg_node.symbol_mapping[str(fsym)] = fsym
                    if fsym not in sdfg.symbols:
                        if fsym in sdfg.parent_sdfg.symbols:
                            sdfg.add_symbol(str(fsym), stype=sdfg.parent_sdfg.symbols[str(fsym)])
                        else:
                            sdfg.add_symbol(str(fsym), stype=dace.dtypes.int32)

        for _, name, array in self._sdfg.arrays_recursive():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.SDFG

        self._sdfg.arg_names = [arg for arg in self.func.__annotations__.keys() if arg != "return"]
        for arg in self._sdfg.arg_names:
            if (
                arg in self.stencil_object.field_info
                and self.stencil_object.field_info[arg] is None
            ):
                shape = tuple(
                    dace.symbolic.symbol(f"__{arg}_{str(axis)}_size")
                    for axis in self.func.__annotations__[arg].axes
                )
                strides = tuple(
                    dace.symbolic.symbol(f"__{arg}_{str(axis)}_stride")
                    for axis in self.func.__annotations__[arg].axes
                )
                self._sdfg.add_array(
                    arg,
                    shape=shape,
                    strides=strides,
                    dtype=dace.typeclass(str(self.func.__annotations__[arg].dtype)),
                )
            if (
                arg in self.stencil_object.parameter_info
                and self.stencil_object.parameter_info[arg] is None
            ):
                self._sdfg.add_symbol(arg, stype=dace.typeclass(self.func.__annotations__[arg]))
        true_args = [
            arg
            for arg in self._sdfg.signature_arglist(with_types=False)
            if not re.match(f"__.*_._stride", arg) and not re.match(f"__.*_._size", arg)
        ]
        assert len(self._sdfg.arg_names) == len(true_args)
        if kwargs.get("use_disk_sdfg", False):
            self._sdfg.save(self.filename)
            print("saved (__sdfg__):", self.filename)
        SDFGWrapper.loaded_compiled_sdfgs[self.filename] = self._sdfg
        self._sdfg.validate()
        return dace.SDFG.from_json(self._sdfg.to_json())

    def __sdfg_signature__(self):
        return ([arg for arg in self.func.__annotations__.keys() if arg != "return"], [])

    def __sdfg_closure__(self, *args, **kwargs):
        return {}


class AxisIndex:
    def __init__(self, axis: str, index: int, offset: int = 0):
        self.axis = axis
        self.index = index
        self.offset = offset

    def __repr__(self):
        return f"AxisIndex(axis={self.axis}, index={self.index}, offset={self.offset})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __str__(self):
        return f"{self.axis}[{self.index}] + {self.offset}"

    def __add__(self, offset: Union[int, "AxisIndex"]):
        if not isinstance(offset, numbers.Integral) and not isinstance(offset, AxisIndex):
            raise TypeError("Offset should be an integer type or axis index")
        if isinstance(offset, AxisIndex):
            if not self.axis == offset.axis:
                raise ValueError("Only AxisIndex with same axis can be added.")
            offset = offset.offset
        if offset == 0:
            return self
        else:
            return AxisIndex(self.axis, self.index, self.offset + offset)

    def __radd__(self, offset: int):
        return self.__add__(offset)

    def __sub__(self, offset: int):
        return self.__add__(-offset)

    def __rsub__(self, offset: int):
        return self.__radd__(-offset)

    def __neg__(self):
        return AxisIndex(self.axis, self.index, -self.offset)


class AxisInterval:
    def __init__(self, axis: str, start: int, end: int):
        assert start < end
        self.axis = axis
        self.start = start
        self.end = end

    def __repr__(self):
        return f"AxisInterval(axis={self.axis}, start={self.start}, end={self.end})"

    def __str__(self):
        return f"{self.axis}[{self.start}:{self.end}]"

    def __len__(self):
        return self.end - self.start


# GTScript builtins: domain axes
class Axis:
    def __init__(self, name: str):
        assert name
        self.name = name

    def __repr__(self):
        return f"Axis(name={self.name})"

    def __str__(self):
        return self.name

    def __getitem__(self, interval):
        if isinstance(interval, slice):
            return AxisInterval(self.name, interval.start, interval.stop)
        elif isinstance(interval, int):
            return AxisIndex(self.name, interval)
        else:
            raise TypeError("Unrecognized index type")


I = Axis("I")
"""I axes (parallel)."""

J = Axis("J")
"""J axes (parallel)."""

K = Axis("K")
"""K axes (sequential)."""

IJ = (I, J)
"""Tuple of axes I, J."""

IK = (I, K)
"""Tuple of axes I, K."""

JK = (J, K)
"""Tuple of axes J, K."""

IJK = (I, J, K)
"""Tuple of axes I, J, K."""


def mask_from_axes(axes):
    if isinstance(axes, Axis):
        axes = (axes,)
    axes = list(a.name for a in axes)
    return list(a in axes for a in list(a.name for a in IJK))


# GTScript builtins: iteration orders
FORWARD = +1
"""Forward iteration order."""

BACKWARD = -1
"""Backward iteration order."""

PARALLEL = 0
"""Parallel iteration order."""


from itertools import count


sym_ctr = count()


class _FieldDescriptor:
    def __init__(self, dtype, axes, data_dims=tuple()):
        if isinstance(dtype, str):
            self.dtype = dtype
        else:
            try:
                dtype = np.dtype(dtype)
                actual_dtype = dtype.subdtype[0] if dtype.subdtype else dtype
                if actual_dtype not in _VALID_DATA_TYPES:
                    raise ValueError("Invalid data type descriptor")
            except:
                raise ValueError("Invalid data type descriptor")
            self.dtype = np.dtype(dtype)
        self.axes = axes if isinstance(axes, collections.abc.Collection) else [axes]
        if data_dims:
            if not isinstance(data_dims, collections.abc.Collection):
                self.data_dims = (data_dims,)
            else:
                self.data_dims = tuple(data_dims)
        else:
            self.data_dims = data_dims

    # def __descriptor__(self):
    #     shape = [dace.symbol(f"__sym_{next(sym_ctr)}_{ax}_size") for ax in self.axes]
    #     strides = [dace.symbol(f"__sym_{next(sym_ctr)}_{ax}_stride") for ax in self.axes]
    #     return dace.data.Array(shape=shape, strides=strides, dtype=dace.typeclass(str(self.dtype)))

    def __repr__(self):
        args = f"dtype={repr(self.dtype)}, axes={repr(self.axes)}, data_dims={repr(self.data_dims)}"
        return f"_FieldDescriptor({args})"

    def __str__(self):
        return (
            f"Field<[{', '.join(str(ax) for ax in self.axes)}], ({self.dtype}, {self.data_dims})>"
        )


class _FieldDescriptorMaker:
    @staticmethod
    def _is_axes_spec(spec) -> bool:
        return (
            isinstance(spec, Axis)
            or isinstance(spec, collections.abc.Collection)
            and all(isinstance(i, Axis) for i in spec)
        )

    def __getitem__(self, field_spec):
        axes = IJK
        data_dims = ()

        if isinstance(field_spec, str) or not isinstance(field_spec, collections.abc.Collection):
            # Field[dtype]
            dtype = field_spec
        elif _FieldDescriptorMaker._is_axes_spec(field_spec[0]):
            # Field[axes, dtype]
            assert len(field_spec) == 2
            axes, dtype = field_spec
        elif len(field_spec) == 2 and not _FieldDescriptorMaker._is_axes_spec(field_spec[1]):
            # Field[high_dimensional_dtype]
            dtype = field_spec
        else:
            raise ValueError("Invalid field type descriptor")

        if isinstance(dtype, collections.abc.Collection) and not isinstance(dtype, str):
            # high dimensional dtype also includes data axes
            assert len(dtype) == 2
            dtype, data_dims = dtype

        return _FieldDescriptor(dtype, axes, data_dims)


# GTScript builtins: variable annotations
Field = _FieldDescriptorMaker()
"""Field descriptor."""


class _SequenceDescriptor:
    def __init__(self, dtype, length):
        self.dtype = dtype
        self.length = length


class _SequenceDescriptorMaker:
    def __getitem__(self, dtype, length=None):
        return dtype, length


Sequence = _SequenceDescriptorMaker()
"""Sequence descriptor."""


# GTScript builtins: external definitions
def externals(*args):
    """Inlined values of the externals definitions."""
    return args


# GTScript builtins: computation and interval statements
class _ComputationContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def computation(order):
    """Define the computation."""
    return _ComputationContextManager()


def interval(*args):
    """Define the interval of computation in the 'K' sequential axis."""
    pass


def horizontal(*args):
    """Define a block of code that is restricted to a set of regions in the parallel axes."""
    pass


class _Region:
    def __getitem__(self, *args):
        """Define a region in the parallel axes."""
        pass


# Horizontal regions
region = _Region()


def index(axis):
    """Current axis index."""
    pass


def range(start, stop):
    """Range from start to stop"""
    pass


def __INLINED(compile_if_expression):
    """Evaluate condition at compile time and inline statements from selected branch."""
    pass


def compile_assert(expr):
    """Assert that expr evaluates to True at compile-time."""
    pass


# GTScript builtins: math functions
def abs(x):
    """Return the absolute value of the argument"""
    pass


def min(x, y):
    """Return the smallest of two or more arguments."""
    pass


def max(x, y):
    """Return the largest of two or more arguments."""
    pass


def mod(x, y):
    """returns the first argument modulo the second one"""
    pass


def sin(x):
    """Return the sine of x radians"""
    pass


def cos(x):
    """Return the cosine of x radians."""
    pass


def tan(x):
    """Return the tangent of x radians."""
    pass


def asin(x):
    """return the arc sine of x, in radians."""
    pass


def acos(x):
    """Return the arc cosine of x, in radians."""
    pass


def atan(x):
    """Return the arc tangent of x, in radians."""
    pass


def sqrt(x):
    """Return the square root of x."""
    pass


def exp(x):
    """Return e raised to the power x, where e is the base of natural logarithms."""
    pass


def log(x):
    """Return the natural logarithm of x (to base e)."""
    pass


def isfinite(x):
    """Return True if x is neither an infinity nor a NaN, and False otherwise. (Note that 0.0 is considered finite.)"""
    pass


def isinf(x):
    """Return True if x is a positive or negative infinity, and False otherwise."""
    pass


def isnan(x):
    """Return True if x is a NaN (not a number), and False otherwise."""
    pass


def floor(x):
    """Return the floor of x, the largest integer less than or equal to x."""
    pass


def ceil(x):
    """Return the ceiling of x, the smallest integer greater than or equal to x."""
    pass


def trunc(x):
    """Return the Real value x truncated to an Integral (usually an integer)"""
    pass
