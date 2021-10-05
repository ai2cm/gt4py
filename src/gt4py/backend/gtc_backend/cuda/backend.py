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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

import gtc.utils as gtc_utils
from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import definitions as gt_definitions
from gt4py import utils as gt_utils
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin, PyExtModuleGenerator, BaseModuleGenerator
from gt4py.backend.gt_backends import (
    GTCUDAPyModuleGenerator,
    cuda_is_compatible_layout,
    cuda_is_compatible_type,
    make_cuda_layout_map,
)
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.cuir import cuir, cuir_codegen, extent_analysis, kernel_fusion, oir_to_cuir, dependency_analysis
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.remove_regions import RemoveUnexecutedRegions
from gtc.passes.oir_pipeline import OirPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCCudaExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        oir_pipeline = OirPipeline(gtir_to_oir.GTIRToOIR().visit(gtir))
        pass_names = self.backend.builder.options.backend_opts.get("skip_passes", ())
        skip_passes = [
            RemoveUnexecutedRegions,
            NoFieldAccessPruning,
        ] + oir_pipeline.steps_from_names(pass_names)
        oir = oir_pipeline.full(skip=skip_passes)

        cuir = oir_to_cuir.OIRToCUIR().visit(oir)
        cuir = kernel_fusion.FuseKernels().visit(cuir)
        cuir = extent_analysis.ComputeExtents().visit(cuir)
        cuir = extent_analysis.CacheExtents().visit(cuir)
        cuir = dependency_analysis.DependencyAnalysis().visit(cuir)

        block_size = self.backend.builder.options.backend_opts.get("block_size", None)
        format_source = self.backend.builder.options.format_source
        implementation = cuir_codegen.CUIRCodegen.apply(
            cuir, block_size=block_size, format_source=format_source
        )

        bindings = GTCCudaBindingsCodegen.apply(
            cuir,
            module_name=self.module_name,
            backend=self.backend,
            format_source=format_source,
        )

        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cu": bindings},
        }


class GTCCudaBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self, backend):
        self.backend = backend
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    def visit_DataType(self, dtype: DataType, **kwargs):
        return cuir_codegen.CUIRCodegen().visit_DataType(dtype)

    def visit_FieldDecl(self, node: cuir.FieldDecl, **kwargs):
        if "external_arg" in kwargs:
            domain_ndim = node.dimensions.count(True)
            data_ndim = len(node.data_dims)
            sid_ndim = domain_ndim + data_ndim
            if kwargs["external_arg"]:
                return "py::buffer {name}, std::array<gt::uint_t,{sid_ndim}> {name}_origin".format(
                    name=node.name,
                    sid_ndim=sid_ndim,
                )
            else:
                return pybuffer_to_sid(
                    name=node.name,
                    ctype=self.visit(node.dtype),
                    domain_dim_flags=node.dimensions,
                    data_ndim=len(node.data_dims),
                    stride_kind_index=self.unique_index(),
                    backend=self.backend,
                )

    def visit_ScalarDecl(self, node: cuir.ScalarDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "{dtype} {name}".format(name=node.name, dtype=self.visit(node.dtype))
            else:
                return "gridtools::stencil::make_global_parameter({name})".format(name=node.name)

    def visit_Program(self, node: cuir.Program, **kwargs):
        assert "module_name" in kwargs
        entry_params = self.visit(node.params, external_arg=True, **kwargs)
        sid_params = self.visit(node.params, external_arg=False, **kwargs)
        return self.generic_visit(
            node,
            entry_params=entry_params,
            sid_params=sid_params,
            **kwargs,
        )

    Program = as_mako(
        """
        #include <chrono>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/cartesian.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;

        extern "C" void run_computation_${name}(
            ${','.join(["std::array<gt::uint_t, 3> domain", *entry_params, 'py::object exec_info'])},
            std::array<int64_t, NUM_KERNELS> streams){
                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
                }

                ${name}(domain)(${','.join(sid_params)}, streams);

                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
                }

            }

        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", run_computation_${name}, "Runs the given computation");

        m.def("num_kernels", []() {
                return NUM_KERNELS;
            }, "Get number of CUDA kernels");

        m.def("has_dependency_info", []() {
                return DEPENDENCY;
            }, "whether or not dependency info is present in the module");

        m.def("dependency_row_ind", []() {
                return DEPENDENCY_ROW_IND;
            }, "Get row ind of dependency matrix stored in csr format");

        m.def("dependency_col_ind", []() {
                return DEPENDENCY_COL_IND;
            }, "Get col ind of dependency matrix stored in csr format");
        }
        """
    )

    @classmethod
    def apply(cls, root, *, module_name="stencil", backend, **kwargs) -> str:
        generated_code = cls(backend).visit(root, module_name=module_name, **kwargs)
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code

class GTCCudaPyModuleGenerator(PyExtModuleGenerator):

    def generate_imports(self) -> str:
        source = """
import cupy
from gt4py import utils as gt_utils
            """
        return source

    def generate_class_members(self) -> str:
        source = ""
        if self.builder.implementation_ir.multi_stages:
            source += """
_pyext_module = gt_utils.make_module_from_file(
    "{pyext_module_name}", "{pyext_file_path}", public_import=True
    )

@property
def pyext_module(self):
    return type(self)._pyext_module
                """.format(
                pyext_module_name=self.pyext_module_name, pyext_file_path=self.pyext_file_path
            )
        return source

    def generate_pre_run(self) -> str:
        field_names = [
            key for key in self.args_data.field_info if self.args_data.field_info[key] is not None
        ]

        return "\n".join([f + ".host_to_device()" for f in field_names])

    def generate_post_run(self) -> str:
        output_field_names = [
            name
            for name, info in self.args_data.field_info.items()
            if info is not None and bool(info.access & gt_definitions.AccessKind.WRITE)
        ]

        return "\n".join([f + "._set_device_modified()" for f in output_field_names])

    def generate_implementation(self) -> str:
        definition_ir = self.builder.definition_ir
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        api_fields = set(field.name for field in definition_ir.api_fields)
        for arg in definition_ir.api_signature:
            if arg.name not in self.args_data.unreferenced:
                args.append(arg.name)
                if arg.name in api_fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self._has_effect():
            source = """
# Load or generate a GTComputation object for the current domain size
num_kernels = self.pyext_module.num_kernels()
if isinstance(streams, int):
    streams = [streams]*num_kernels
self.pyext_module.run_computation(list(_domain_), {run_args}, exec_info, list(streams))
""".format(
                run_args=", ".join(args)
            )
            sources.extend(source.splitlines())
        else:
            sources.extend("\n")

        return sources.text+"""
if not async_launch: cupy.cuda.Device(0).synchronize()
        """

@gt_backend.register
class GTCCudaBackend(BaseGTBackend, CLIBackendMixin):
    """CUDA backend using gtc."""

    name = "gtc:cuda"
    options = {
        **BaseGTBackend.GT_BACKEND_OPTS,
        "device_sync": {"versioning": True, "type": bool},
        "block_size": {"versioning": True, "type": Tuple[int, int]},
    }
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }
    PYEXT_GENERATOR_CLASS = GTCCudaExtGenerator  # type: ignore
    MODULE_GENERATOR_CLASS = GTCCudaPyModuleGenerator
    GT_BACKEND_T = "gpu"

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=True)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO(havogt) add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )
