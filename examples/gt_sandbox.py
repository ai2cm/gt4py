#!/usr/bin/env python

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import sys
import time
import warnings
from types import SimpleNamespace
from typing import Any, Sequence

warnings.filterwarnings(
    action="ignore",
    module=r".*dace",
)

import numpy as np

# from fv3core.stencils.ytp_v import _ytp_v
# from fv3core.stencils.yppm import compute_y_flux
# from fv3core.stencils.xppm import compute_x_flux
from fv3core.decorators import FrozenStencil, computepath_method
from fv3core.stencils.nh_p_grad import set_k0_and_calc_wk
from fv3core.stencils.sim1_solver import sim1_solver
from fv3core.stencils.c_sw import update_x_velocity, update_y_velocity, correct_y_edge_velocity, correct_x_edge_velocity

# from fv3core.utils.corners import fill_corners_bgrid_x_defn

# from fv3core.stencils.map_single import lagrangian_contributions
# from fv3core.utils.corners import copy_corners_x_stencil_defn
# from fv3core.stencils.delnflux import copy_corners_x_nord
from fv3core.utils.global_config import set_backend, set_rebuild
from fv3core.utils.mpi import MPI
from fv3core.utils.typing import BoolField, FloatField, FloatFieldIJ

from fv3gfs.util import Quantity

try:
    # from fv3core.utils.future_stencil import future_stencil
    future_stencil = None
except ModuleNotFoundError:
    future_stencil = None

import gt4py as gt
import gt4py.storage as gt_storage
from gt4py.gtscript import (
    FORWARD,
    IJ,
    IJK,
    PARALLEL,
    Field,
    I,
    J,
    K,
    computation,
    horizontal,
    interval,
    region,
    stencil,
)

from gtc.passes.oir_pipeline import DefaultPipeline, OirPipeline


# gt_backend = "gtx86"  # "gtx86"
# gt_backend = "gtc:cuda"
# gt_backend = "gtc:numpy"
# gt_backend = "gtc:gt:cpu_ifirst"
gt_backend = "gtc:dace"
# gt_backend = "gtc:dace:gpu"
# gt_backend = "gtc:gt:cpu_ifirst"
# gt_backend = "gtc:gt:gpu"
# gt_backend = "gtx86"
np_backend = "numpy"
# np_backend = "gtx86"


def mask_from_shape(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (False, False, True)
    return (True,) * len(shape) + (False,) * (3 - len(shape))


def arrays_to_storages(array_dict: dict, backend: str, origin: tuple) -> dict:
    return {
        name: gt_storage.from_array(
            data=item,
            backend=backend,
            default_origin=origin,
            shape=item.shape,
            mask=mask_from_shape(item.shape),
            managed_memory=True,
        )
        if item.shape
        else item[()]
        for name, item in array_dict.items()
    }


def double_data(input_data):
    output_data = {}
    for name, item in input_data.items():
        if item.shape:
            array = item
            array2 = np.concatenate((array, array), axis=1)
            array3 = np.concatenate((array2, array2), axis=0)
            item = array3
        output_data[name] = item
    return output_data


def get_pass_pipeline(skip_passes: Sequence[str]) -> Any:  # OirPipeline:
    step_map = {step.__name__: step for step in DefaultPipeline.all_steps()}
    skip_steps = [step_map[pass_name] for pass_name in skip_passes]
    return DefaultPipeline(skip=skip_steps)


# @gtscript.stencil(backend=backend)
# def part5(
#     chz: FIELD_FLT,
#     ckz: FIELD_FLT,
#     hpbl: FIELD_FLT,
#     kpbl: FIELD_INT,
#     mask: FIELD_INT,
#     pcnvflg: FIELD_BOOL,
#     phih: FIELD_FLT,
#     phim: FIELD_FLT,
#     prn: FIELD_FLT,
#     zi: FIELD_FLT,
# ):

#     with computation(FORWARD), interval(1, None):
#         phih = phih[0, 0, -1]
#         phim = phim[0, 0, -1]

#     with computation(PARALLEL), interval(...):0
#         tem1 = max(zi[0, 0, 1] - sfcfrac * hpbl[0, 0, 0], 0.0)
#         ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0, 0] ** 2.0)
#         if mask[0, 0, 0] < kpbl[0, 0, 0]:
#             if pcnvflg[0, 0, 0]:
#                 prn = 1.0 + ((phih[0, 0, 0] / phim[0, 0, 0]) - 1.0) * exp(ptem)
#             else:
#                 prn = phih[0, 0, 0] / phim[0, 0, 0]

#         if mask[0, 0, 0] < kpbl[0, 0, 0]:
#             prn = max(min(prn[0, 0, 0], prmax), prmin)
#             ckz = max(min(ck1 + (ck0 - ck1) * exp(ptem), ck0), ck1)
#             chz = max(min(ch1 + (ch0 - ch1) * exp(ptem), ch0), ch1)


class VorticityTransport:
    def __init__(self, **kwargs):
        origin = (3, 3, 0)
        domain = (13, 13, 79)

        set_backend(kwargs.pop("backend"))
        set_rebuild(kwargs.pop("rebuild"))

        # ke_c: kinetic energy on C-grid (input)
        self._ke = kwargs.pop("ke")
        # vort_c: Vorticity on C-grid (input)
        self._vort = kwargs.pop("vort")

        self.grid = SimpleNamespace(
            south_edge=True,
            north_edge=True,
            west_edge=True,
            east_edge=True,
            cosa_u=kwargs.pop("cosa_u"),
            cosa_v=kwargs.pop("cosa_v"),
            sina_u=kwargs.pop("sina_u"),
            sina_v=kwargs.pop("sina_v"),
            rdxc=kwargs.pop("rdxc"),
            rdyc=kwargs.pop("rdyc"),
            js=origin[1],
            je=domain[1] + 1,
            is_=origin[0],
            ie=domain[0] + 1,
            nic=domain[0] - 1,
            njc=domain[1] - 1,
            npz=domain[2],
        )

        js = self.grid.js + 1 if self.grid.south_edge else self.grid.js
        je = self.grid.je if self.grid.north_edge else self.grid.je + 1
        self._update_y_velocity = FrozenStencil(
            func=update_y_velocity,
            origin=(self.grid.is_, js, 0),
            domain=(self.grid.nic, je - js + 1, self.grid.npz),
        )
        self._update_south_velocity = FrozenStencil(
            correct_y_edge_velocity,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(self.grid.nic, 1, self.grid.npz),
        )
        self._update_north_velocity = FrozenStencil(
            correct_y_edge_velocity,
            origin=(self.grid.is_, self.grid.je + 1, 0),
            domain=(self.grid.nic, 1, self.grid.npz),
        )
        is_ = self.grid.is_ + 1 if self.grid.west_edge else self.grid.is_
        ie = self.grid.ie if self.grid.east_edge else self.grid.ie + 1
        self._update_x_velocity = FrozenStencil(
            func=update_x_velocity,
            origin=(is_, self.grid.js, 0),
            domain=(ie - is_ + 1, self.grid.njc, self.grid.npz),
        )
        self._update_west_velocity = FrozenStencil(
            correct_x_edge_velocity,
            origin=(self.grid.is_, self.grid.js, 0),
            domain=(1, self.grid.njc, self.grid.npz),
        )
        self._update_east_velocity = FrozenStencil(
            correct_x_edge_velocity,
            origin=(self.grid.ie + 1, self.grid.js, 0),
            domain=(1, self.grid.njc, self.grid.npz),
        )

    @computepath_method
    def __call__(self, **kwargs):
        self._vorticitytransport_cgrid(**kwargs)

    @computepath_method
    def _vorticitytransport_cgrid(
        self,
        uc,
        vc,
        v,
        u,
        dt2: float,
    ):
        """Update the C-Grid x and y velocity fields.

        Args:
            uc: x-velocity on C-grid (input, output)
            vc: y-velocity on C-grid (input, output)
            v: y-velocity on D-grid (input)
            u: x-velocity on D-grid (input)
            dt2: timestep (input)
        """
        self._update_y_velocity(
            self._vort,
            self._ke,
            u,
            vc,
            self.grid.cosa_v,
            self.grid.sina_v,
            self.grid.rdyc,
            dt2,
        )
        if self.grid.south_edge:
            self._update_south_velocity(
                self._vort,
                self._ke,
                u,
                vc,
                self.grid.rdyc,
                dt2,
            )
        if self.grid.north_edge:
            self._update_north_velocity(
                self._vort,
                self._ke,
                u,
                vc,
                self.grid.rdyc,
                dt2,
            )
        self._update_x_velocity(
            self._vort,
            self._ke,
            v,
            uc,
            self.grid.cosa_u,
            self.grid.sina_u,
            self.grid.rdxc,
            dt2,
        )
        if self.grid.west_edge:
            self._update_west_velocity(
                self._vort,
                self._ke,
                v,
                uc,
                self.grid.rdxc,
                dt2,
            )
        if self.grid.west_edge:
            self._update_east_velocity(
                self._vort,
                self._ke,
                v,
                uc,
                self.grid.rdxc,
                dt2,
            )


def main():
    hsize = 18
    vsize = hsize + 1
    nhalo = 3  # 2  # 1

    # definition_func = lagrangian_contributions
    # definition_func = sim1_solver
    # definition_func = fill_corners_bgrid_x_defn
    # definition_func = set_k0_and_calc_wk
    definition_func = update_x_velocity
    # definition_func = for_loop_test

    # nonhydrostatic_pressure_gradient = (
    #     NonHydrostaticPressureGradient(grid_type=0)
    # )

    data_file_prefix = sys.argv[1] if len(sys.argv) > 1 else definition_func.__name__
    is_parallel: bool = MPI is not None and MPI.COMM_WORLD.Get_size() > 1
    data_file_prefix += f"_r{MPI.COMM_WORLD.Get_rank()}" if is_parallel else ""
    input_data = dict(np.load(f"{data_file_prefix}.npz", allow_pickle=True))

    origin = tuple(input_data.pop("origin", []))
    if not any(origin):
        origin = (nhalo, nhalo, 0)

    domain = tuple(input_data.pop("domain", []))
    if not any(domain):
        domain = (hsize, hsize, vsize)

    externals = input_data.pop("externals", None)
    if externals:
        externals = externals[()]
        axes = {"I": I, "J": J}
        for name, value in externals.items():
            if isinstance(value, dict) and "axis" in value:
                axis_index = axes[value["axis"]][value["index"]] + value["offset"]
                assert axis_index.__dict__ == value
                externals[name] = axis_index
    else:
        externals = {}

    # skip_passes = ("GreedyMerging",)
    skip_passes = ("graph_merge_horizontal_executions",)
    stencil_function = future_stencil if future_stencil and MPI else stencil

    do_rebuild: bool = True
    build_info: dict = {}
    gt_stencil = stencil_function(
        definition=definition_func,
        backend=gt_backend,
        externals=externals,
        rebuild=do_rebuild,
        oir_pipeline=get_pass_pipeline(skip_passes),
        # disable_code_generation=True,
        build_info=build_info,
    )
    # field_info = gt_stencil.field_info
    np_stencil = stencil_function(
        definition=definition_func,
        backend=np_backend,
        externals=externals,
        rebuild=do_rebuild,
    )

    if build_info:
        if "codegen_time" in build_info:
            print(
                f"compile_time (backend={gt_backend}): codegen_time={build_info['codegen_time']}, build_time={build_info['build_time']}"
            )
        elif "load_time" in build_info:
            print(
                f"compile_time (backend={gt_backend}): parse_time={build_info['parse_time']}, load_time={build_info['load_time']}"
            )

    add_2d_temp: bool = False
    if add_2d_temp:
        input_data["tmp"] = np.zeros(input_data["w"].shape[0:2])

    n_doubles: int = 0
    if n_doubles:
        for _ in range(n_doubles):
            input_data = double_data(input_data)
            domain = (domain[0] * 2, domain[1] * 2, domain[2])

    gt_inputs = arrays_to_storages(input_data, gt_backend, origin)
    np_inputs = arrays_to_storages(input_data, np_backend, origin)

    skip_fields = []  # ("q4_2", "q4_3", "q4_4")
    for field in skip_fields:
        gt_inputs.pop(field)
        np_inputs.pop(field)

    call_args = ("uc", "vc", "v", "u", "dt2")
    gt_call_kwargs = {arg: gt_inputs.pop(arg) for arg in call_args}
    vort_xport_gt = VorticityTransport(backend=gt_backend, rebuild=do_rebuild, **gt_inputs)
    vort_xport_gt(**gt_call_kwargs)

    np_call_kwargs = {arg: np_inputs.pop(arg) for arg in call_args}
    vort_xport_np = VorticityTransport(backend=np_backend, rebuild=do_rebuild, **np_inputs)
    vort_xport_np(**np_call_kwargs)

    for name, field in gt_call_kwargs.items():
        array = np.asarray(field)
        assert np.allclose(np_call_kwargs[name], array, equal_nan=True)

    # Invoke numpy stencil...
    np_stencil(domain=domain, origin=origin, **np_inputs)

    # TODO: Automate this...
    # gt_domain = (1, 1, domain[2])
    gt_domain = domain

    n_runs: int = 1
    total_time: float = 0.0
    for _ in range(n_runs):
        exec_info = {}
        run_time: float = time.perf_counter()
        gt_stencil(domain=gt_domain, origin=origin, exec_info=exec_info, **gt_inputs)
        if "run_cpp_end_time" in exec_info:
            run_time = exec_info["run_cpp_end_time"] - exec_info["run_cpp_start_time"]
        else:
            run_time = time.perf_counter() - run_time
        total_time += run_time * 1e3
    mean_time = total_time / float(n_runs)
    print(
        f"mean_time (backend={gt_backend}, domain={domain}, n_runs={n_runs}) = {mean_time} ms"
    )

    fail_arrays = {}
    for name in gt_inputs.keys():
        if isinstance(gt_inputs[name], gt_storage.Storage):
            gt_array = np.asarray(gt_inputs[name])
            np_array = np.asarray(np_inputs[name])
            if not np.allclose(gt_array, np_array, equal_nan=True):
                fail_arrays[name] = (gt_array, np_array)

    if fail_arrays:
        for array_tuple in fail_arrays.values():
            gt_array, np_array = array_tuple
            diff_array = gt_array - np_array
            diff_indices = np.transpose(diff_array[:, :, 0].nonzero())
            fail_ratio = diff_indices.shape[0] / (
                diff_array.shape[0] * diff_array.shape[1]
            )

            # print(f"np_array = {np_array[:, :, 0]}")
            # print(f"gt_array = {gt_array[:, :, 0]}")
            # print(f"diff_array = {diff_array[:, :, 0]}")
            # print(f"diff_indices = {diff_indices}")
            print(f"fail_ratio = {fail_ratio}")
        print(f"{fail_arrays.keys()} fail")
    else:
        print("All good!")


if __name__ == "__main__":
    main()
