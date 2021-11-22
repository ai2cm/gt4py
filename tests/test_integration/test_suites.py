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


import numpy as np
import pytest

from gt4py import gtscript
from gt4py import testing as gt_testing
from gt4py.gtscript import PARALLEL, computation, interval

from ..definitions import INTERNAL_BACKENDS
from .stencil_definitions import optional_field, two_optional_fields


# ---- Identity stencil ----
class TestIdentity(gt_testing.StencilTestSuite):
    """Identity stencil."""

    dtypes = {("field_a",): (np.float64, np.float32)}
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = INTERNAL_BACKENDS
    symbols = dict(field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(field_a):
        with computation(PARALLEL), interval(...):
            tmp = field_a
            field_a = tmp

    def validation(field_a, domain=None, origin=None):
        pass


# ---- Copy stencil ----
class TestCopy(gt_testing.StencilTestSuite):
    """Copy stencil."""

    dtypes = (np.float_,)
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b):
        with computation(PARALLEL), interval(...):
            field_b = field_a  # noqa: F841  # Local name is assigned to but never used

    def validation(field_a, field_b, domain=None, origin=None):
        field_b[...] = field_a


class TestAugAssign(gt_testing.StencilTestSuite):
    """Increment by one stencil."""

    dtypes = (np.float_,)
    domain_range = [(1, 25), (1, 25), (1, 25)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b):
        with computation(PARALLEL), interval(...):
            field_a += 1.0
            field_a *= 2.0
            field_b -= 1.0
            field_b /= 2.0

    def validation(field_a, field_b, domain=None, origin=None):
        field_a[...] = (field_a[...] + 1.0) * 2.0
        field_b[...] = (field_b[...] - 1.0) / 2.0


# ---- Scale stencil ----
class TestGlobalScale(gt_testing.StencilTestSuite):
    """Scale stencil using a global global_name."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        SCALE_FACTOR=gt_testing.global_name(one_of=(1.0, 1e3, 1e6)),
        field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a):
        from __externals__ import SCALE_FACTOR

        with computation(PARALLEL), interval(...):
            field_a = SCALE_FACTOR * field_a[0, 0, 0]

    def validation(field_a, domain, origin, **kwargs):
        field_a[...] = SCALE_FACTOR * field_a  # noqa: F821  # Undefined name


# ---- Parametric scale stencil -----
class TestParametricScale(gt_testing.StencilTestSuite):
    """Scale stencil using a parameter."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        scale=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(field_a, *, scale):
        with computation(PARALLEL), interval(...):
            field_a = scale * field_a

    def validation(field_a, *, scale, domain, origin, **kwargs):
        field_a[...] = scale * field_a


# --- Parametric-mix stencil ----
class TestParametricMix(gt_testing.StencilTestSuite):
    """Linear combination of input fields using several parameters."""

    dtypes = {
        ("USE_ALPHA",): np.int_,
        ("field_a", "field_b", "field_c"): np.float64,
        ("field_out",): np.float32,
        ("weight", "alpha_factor"): np.float_,
    }
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        USE_ALPHA=gt_testing.global_name(one_of=(True, False)),
        field_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_c=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_out=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(-10, 10)),
        alpha_factor=gt_testing.parameter(in_range=(-1, 1)),
    )

    def definition(field_a, field_b, field_c, field_out, *, weight, alpha_factor):
        from __externals__ import USE_ALPHA
        from __gtscript__ import __INLINED

        with computation(PARALLEL), interval(...):
            if __INLINED(USE_ALPHA):
                factor = alpha_factor
            else:
                factor = 1.0
            field_out = factor * field_a[  # noqa: F841 # Local name is assigned to but never used
                0, 0, 0
            ] - (1 - factor) * (field_b[0, 0, 0] - weight * field_c[0, 0, 0])

    def validation(
        field_a, field_b, field_c, field_out, *, weight, alpha_factor, domain, origin, **kwargs
    ):
        if USE_ALPHA:  # noqa: F821  # Undefined name
            factor = alpha_factor
        else:
            factor = 1.0
        field_out[...] = (factor * field_a[:, :, :]) - (1 - factor) * (
            field_b[:, :, :] - (weight * field_c[:, :, :])
        )


class TestHeatEquation_FTCS_3D(gt_testing.StencilTestSuite):
    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        u=gt_testing.field(in_range=(-10, 10), extent=[(-1, 1), (0, 0), (0, 0)]),
        v=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (-1, 1), (0, 0)]),
        u_new=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (0, 0), (0, 0)]),
        v_new=gt_testing.field(in_range=(-10, 10), extent=[(0, 0), (0, 0), (0, 0)]),
        ru=gt_testing.parameter(in_range=(0, 0.5)),
        rv=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, v, u_new, v_new, *, ru, rv):
        with computation(PARALLEL), interval(...):
            u_new = u[0, 0, 0] + ru * (  # noqa: F841 # Local name is assigned to but never used
                u[1, 0, 0] - 2 * u[0, 0, 0] + u[-1, 0, 0]
            )
            v_new = v[0, 0, 0] + rv * (  # noqa: F841 # Local name is assigned to but never used
                v[0, 1, 0] - 2 * v[0, 0, 0] + v[0, -1, 0]
            )

    def validation(u, v, u_new, v_new, *, ru, rv, domain, origin, **kwargs):
        u_new[...] = u[1:-1, :, :] + ru * (u[2:, :, :] - 2 * u[1:-1, :, :] + u[:-2, :, :])
        v_new[...] = v[:, 1:-1, :] + rv * (v[:, 2:, :] - 2 * v[:, 1:-1, :] + v[:, :-2, :])


class TestHorizontalDiffusion(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        with computation(PARALLEL), interval(...):
            laplacian = 4.0 * u[0, 0, 0] - (u[1, 0, 0] + u[-1, 0, 0] + u[0, 1, 0] + u[0, -1, 0])
            flux_i = laplacian[1, 0, 0] - laplacian[0, 0, 0]
            flux_j = laplacian[0, 1, 0] - laplacian[0, 0, 0]
            diffusion = u[  # noqa: F841 # Local name is assigned to but never used
                0, 0, 0
            ] - weight * (flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0])

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


@gtscript.function
def lap_op(u):
    """Laplacian operator."""
    return 4.0 * u[0, 0, 0] - (u[1, 0, 0] + u[-1, 0, 0] + u[0, 1, 0] + u[0, -1, 0])


@gtscript.function
def fwd_diff_op_xy(field):
    dx = field[1, 0, 0] - field[0, 0, 0]
    dy = field[0, 1, 0] - field[0, 0, 0]
    return dx, dy


@gtscript.function
def wrap1arg2return(field):
    dx, dy = fwd_diff_op_xy(field=field)
    return dx, dy


@gtscript.function
def fwd_diff_op_x(field):
    dx = field[1, 0, 0] - field[0, 0, 0]
    return dx


@gtscript.function
def fwd_diff_op_y(field):
    dy = field[0, 1, 0] - field[0, 0, 0]
    return dy


class TestHorizontalDiffusionSubroutines(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        fwd_diff=gt_testing.global_name(singleton=wrap1arg2return),
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        from __externals__ import fwd_diff

        with computation(PARALLEL), interval(...):
            laplacian = lap_op(u=u)
            flux_i, flux_j = fwd_diff(field=laplacian)
            diffusion = u[  # noqa: F841 # Local name is assigned to but never used
                0, 0, 0
            ] - weight * (flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0])

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


class TestHorizontalDiffusionSubroutines2(gt_testing.StencilTestSuite):
    """Diffusion in a horizontal 2D plane ."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        fwd_diff=gt_testing.global_name(singleton=fwd_diff_op_xy),
        BRANCH=gt_testing.global_name(one_of=(True, False)),
        u=gt_testing.field(in_range=(-10, 10), boundary=[(2, 2), (2, 2), (0, 0)]),
        diffusion=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        weight=gt_testing.parameter(in_range=(0, 0.5)),
    )

    def definition(u, diffusion, *, weight):
        from __externals__ import BRANCH
        from __gtscript__ import __INLINED

        with computation(PARALLEL), interval(...):
            laplacian = lap_op(u=u)
            if __INLINED(BRANCH):
                flux_i = fwd_diff_op_x(field=laplacian)
                flux_j = fwd_diff_op_y(field=laplacian)
            else:
                flux_i, flux_j = fwd_diff_op_xy(field=laplacian)
            diffusion = u[  # noqa: F841 # Local name is assigned to but never used
                0, 0, 0
            ] - weight * (flux_i[0, 0, 0] - flux_i[-1, 0, 0] + flux_j[0, 0, 0] - flux_j[0, -1, 0])

    def validation(u, diffusion, *, weight, domain, origin, **kwargs):
        laplacian = 4.0 * u[1:-1, 1:-1, :] - (
            u[2:, 1:-1, :] + u[:-2, 1:-1, :] + u[1:-1, 2:, :] + u[1:-1, :-2, :]
        )
        flux_i = laplacian[1:, 1:-1, :] - laplacian[:-1, 1:-1, :]
        flux_j = laplacian[1:-1, 1:, :] - laplacian[1:-1, :-1, :]
        diffusion[...] = u[2:-2, 2:-2, :] - weight * (
            flux_i[1:, :, :] - flux_i[:-1, :, :] + flux_j[:, 1:, :] - flux_j[:, :-1, :]
        )


class TestRuntimeIfFlat(gt_testing.StencilTestSuite):
    """Tests runtime ifs."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(outfield):

        with computation(PARALLEL), interval(...):

            if True:
                outfield = 1
            else:
                outfield = 2  # noqa: F841  # Local name is assigned to but never used

    def validation(outfield, *, domain, origin, **kwargs):
        outfield[...] = 1


class TestRuntimeIfNested(gt_testing.StencilTestSuite):
    """Tests nested runtime ifs."""

    dtypes = (np.float_,)
    domain_range = [(1, 15), (1, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(outfield):

        with computation(PARALLEL), interval(...):
            if (outfield > 0 and outfield > 0) or (not outfield > 0 and not outfield > 0):
                if False:
                    outfield = 1
                else:
                    outfield = 2
            else:
                outfield = 3

    def validation(outfield, *, domain, origin, **kwargs):
        outfield[...] = 2


@gtscript.function
def add_one(field_in):
    """Add 1 to each element of `field_in`."""
    return field_in + 1


class Test3FoldNestedIf(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(3, 3), (3, 3), (3, 3)]
    backends = INTERNAL_BACKENDS
    symbols = dict(field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]))

    def definition(field_a):
        with computation(PARALLEL), interval(...):
            if field_a >= 0.0:
                field_a = 0.0
                if field_a > 1:
                    field_a = 1
                    if field_a > 2:
                        field_a = 2

    def validation(field_a, domain, origin):
        for v in range(3):
            field_a[np.where(field_a > v)] = v


class TestRuntimeIfNestedDataDependent(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(3, 3), (3, 3), (3, 3)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        factor=gt_testing.parameter(in_range=(-100, 100)),
        field_a=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_b=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
        field_c=gt_testing.field(in_range=(-1, 1), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(field_a, field_b, field_c, *, factor):
        with computation(PARALLEL), interval(...):
            if factor > 0:
                if field_a < 0:
                    field_b = -field_a
                else:
                    field_b = field_a  # noqa: F841  # Local name is assigned to but never used
            else:
                if field_a < 0:
                    field_c = -field_a
                else:
                    field_c = field_a  # noqa: F841  # Local name is assigned to but never used

            field_a = add_one(field_a)

    def validation(field_a, field_b, field_c, *, factor, domain, origin, **kwargs):

        if factor > 0:
            field_b[...] = np.abs(field_a)
        else:
            field_c[...] = np.abs(field_a)
        field_a += 1


class TestTernaryOp(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        infield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 1), (0, 0)]),
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(infield, outfield):

        with computation(PARALLEL), interval(...):
            outfield = (  # noqa: F841 # Local name is assigned to but never used
                infield if infield > 0.0 else -infield[0, 1, 0]
            )

    def validation(infield, outfield, *, domain, origin, **kwargs):
        outfield[...] = (infield[:, :-1, :] > 0.0) * infield[:, :-1, :] + (
            infield[:, :-1, :] <= 0.0
        ) * (-infield[:, 1:, :])


class TestThreeWayAnd(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        a=gt_testing.parameter(in_range=(-100, 100)),
        b=gt_testing.parameter(in_range=(-100, 100)),
        c=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(outfield, *, a, b, c):

        with computation(PARALLEL), interval(...):
            if a > 0 and b > 0 and c > 0:
                outfield = 1
            else:
                outfield = 0  # noqa: F841  # Local name is assigned to but never used

    def validation(outfield, *, a, b, c, domain, origin, **kwargs):
        outfield[...] = 1 if a > 0 and b > 0 and c > 0 else 0


class TestThreeWayOr(gt_testing.StencilTestSuite):

    dtypes = (np.float_,)
    domain_range = [(1, 15), (2, 15), (1, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        a=gt_testing.parameter(in_range=(-100, 100)),
        b=gt_testing.parameter(in_range=(-100, 100)),
        c=gt_testing.parameter(in_range=(-100, 100)),
    )

    def definition(outfield, *, a, b, c):

        with computation(PARALLEL), interval(...):
            if a > 0 or b > 0 or c > 0:
                outfield = 1
            else:
                outfield = 0  # noqa: F841  # Local name is assigned to but never used

    def validation(outfield, *, a, b, c, domain, origin, **kwargs):
        outfield[...] = 1 if a > 0 or b > 0 or c > 0 else 0


class TestOptionalField(gt_testing.StencilTestSuite):
    dtypes = (np.float_,)
    domain_range = [(1, 32), (1, 32), (1, 32)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        PHYS_TEND=gt_testing.global_name(one_of=(False, True)),
        in_field=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        out_field=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        dyn_tend=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        phys_tend=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        dt=gt_testing.parameter(in_range=(0, 100)),
    )

    definition = optional_field

    def validation(in_field, out_field, dyn_tend, phys_tend=None, *, dt, domain, origin, **kwargs):

        out_field[...] = in_field + dt * dyn_tend
        if PHYS_TEND:  # noqa: F821  # Undefined name
            out_field += dt * phys_tend


class TestNotSpecifiedOptionalField(TestOptionalField):
    backends = INTERNAL_BACKENDS
    symbols = TestOptionalField.symbols.copy()
    symbols["PHYS_TEND"] = gt_testing.global_name(one_of=(False,))
    symbols["phys_tend"] = gt_testing.none()


class TestTwoOptionalFields(gt_testing.StencilTestSuite):
    dtypes = (np.float_,)
    domain_range = [(1, 32), (1, 32), (1, 32)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        PHYS_TEND_A=gt_testing.global_name(one_of=(False, True)),
        PHYS_TEND_B=gt_testing.global_name(one_of=(False, True)),
        in_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        in_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        out_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        out_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        dyn_tend_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        dyn_tend_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        phys_tend_a=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        phys_tend_b=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        dt=gt_testing.parameter(in_range=(0, 100)),
    )

    definition = two_optional_fields

    def validation(
        in_a,
        in_b,
        out_a,
        out_b,
        dyn_tend_a,
        dyn_tend_b,
        phys_tend_a=None,
        phys_tend_b=None,
        *,
        dt,
        domain,
        origin,
        **kwargs,
    ):

        out_a[...] = in_a + dt * dyn_tend_a
        out_b[...] = in_b + dt * dyn_tend_b
        if PHYS_TEND_A:  # noqa: F821  # Undefined name
            out_a += dt * phys_tend_a
        if PHYS_TEND_B:  # noqa: F821  # Undefined name
            out_b += dt * phys_tend_b


class TestNotSpecifiedTwoOptionalFields(TestTwoOptionalFields):
    backends = INTERNAL_BACKENDS
    symbols = TestTwoOptionalFields.symbols.copy()
    symbols["PHYS_TEND_A"] = gt_testing.global_name(one_of=(False,))
    symbols["phys_tend_a"] = gt_testing.none()


class TestConstantFolding(gt_testing.StencilTestSuite):
    dtypes = {("outfield",): np.float64, ("cond",): np.float64}
    domain_range = [(15, 15), (15, 15), (15, 15)]
    backends = INTERNAL_BACKENDS
    symbols = dict(
        outfield=gt_testing.field(in_range=(-10, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
        cond=gt_testing.field(in_range=(1, 10), boundary=[(0, 0), (0, 0), (0, 0)]),
    )

    def definition(outfield, cond):
        with computation(PARALLEL), interval(...):
            if cond != 0:
                tmp = 1
            outfield = tmp  # noqa: F841  # local variable assigned to but never used

    def validation(outfield, cond, *, domain, origin, **kwargs):
        outfield[np.array(cond, dtype=np.bool_)] = 1


class TestNon3DFields(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "another_field": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 10), (4, 10), (4, 10)]
    backends = [
        "debug",
        "numpy",
        pytest.param("gtx86", marks=[pytest.mark.xfail]),
        pytest.param("gtmc", marks=[pytest.mark.xfail]),
        pytest.param("gtcuda", marks=[pytest.mark.xfail]),
        "gtc:gt:cpu_ifirst",
        "gtc:gt:cpu_kfirst",
        "gtc:gt:gpu",
        "gtc:dace",
    ]
    symbols = {
        "field_in": gt_testing.field(
            in_range=(-10, 10), axes="K", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "another_field": gt_testing.field(
            in_range=(-10, 10), axes="IJ", data_dims=(3, 2, 2), boundary=[(1, 1), (1, 1), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(-10, 10), axes="IJK", data_dims=(3, 2), boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, another_field, field_out):
        with computation(PARALLEL), interval(...):
            field_out[0, 0, 0][0, 0] = (
                field_in[0] + another_field[-1, -1][0, 0, 0] + another_field[-1, -1][0, 0, 1]
            )
            field_out[0, 0, 0][0, 1] = 2 * (
                another_field[-1, -1][1, 0, 0]
                + another_field[-1, -1][1, 0, 1]
                + another_field[-1, -1][1, 1, 0]
                + another_field[-1, -1][1, 1, 1]
            )

            field_out[0, 0, 0][1, 0] = (
                field_in[0] + another_field[1, 1][0, 0, 0] + another_field[1, 1][0, 0, 1]
            )
            field_out[0, 0, 0][1, 1] = 3 * (
                another_field[1, 1][1, 0, 0]
                + another_field[1, 1][1, 0, 1]
                + another_field[1, 1][1, 1, 0]
                + another_field[1, 1][1, 1, 1]
            )

            field_out[0, 0, 0][2, 0] = (
                field_in[0] + another_field[0, 0][0, 0, 0] + another_field[-1, 1][0, 0, 1]
            )
            field_out[0, 0, 0][2, 1] = 4 * (
                another_field[-1, 1][1, 0, 0]
                + another_field[-1, 1][1, 0, 1]
                + another_field[-1, 1][1, 1, 0]
                + another_field[-1, 1][1, 1, 1]
            )

    def validation(field_in, another_field, field_out, *, domain, origin):
        field_out[:, :, :, 0, 0] = (
            field_in[:]
            + another_field[:-2, :-2, None, 0, 0, 0]
            + another_field[:-2, :-2, None, 0, 0, 1]
        )
        field_out[:, :, :, 0, 1] = 2 * (
            another_field[:-2, :-2, None, 1, 0, 0]
            + another_field[:-2, :-2, None, 1, 0, 1]
            + another_field[:-2, :-2, None, 1, 1, 0]
            + another_field[:-2, :-2, None, 1, 1, 1]
        )

        field_out[:, :, :, 1, 0] = (
            field_in[:]
            + another_field[2:, 2:, None, 0, 0, 0]
            + another_field[2:, 2:, None, 0, 0, 1]
        )
        field_out[:, :, :, 1, 1] = 3 * (
            another_field[2:, 2:, None, 1, 0, 0]
            + another_field[2:, 2:, None, 1, 0, 1]
            + another_field[2:, 2:, None, 1, 1, 0]
            + another_field[2:, 2:, None, 1, 1, 1]
        )

        field_out[:, :, :, 2, 0] = (
            field_in[:]
            + another_field[1:-1, 1:-1, None, 0, 0, 0]
            + another_field[:-2, 2:, None, 0, 0, 1]
        )
        field_out[:, :, :, 2, 1] = 4 * (
            another_field[:-2, 2:, None, 1, 0, 0]
            + another_field[:-2, 2:, None, 1, 0, 1]
            + another_field[:-2, 2:, None, 1, 1, 0]
            + another_field[:-2, 2:, None, 1, 1, 1]
        )


class TestReadOutsideKInterval(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "field_in": gt_testing.field(
            in_range=(-10, 10), axes="IJK", boundary=[(0, 0), (0, 0), (1, 1)]
        ),
        "field_out": gt_testing.field(
            in_range=(-10, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            field_out = (  # noqa: F841  # Local name is assigned to but never used
                field_in[0, 0, -1] + field_in[0, 0, 1]
            )

    def validation(field_in, field_out, *, domain, origin):
        field_out[:, :, :] = field_in[:, :, 0:-2] + field_in[:, :, 2:]


Iend = gtscript.AxisIndex(axis=gtscript.I, index=-1, offset=1)
Jend = gtscript.AxisIndex(axis=gtscript.J, index=-1, offset=1)


class TestRegionFullDomain(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(1, 0), (0, 1), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[0:Iend, 0:Jend]):
                field_out = (  # noqa: F841  # Local name is assigned to but never used
                    field_in[-1, 0, 0] + field_in[0, 1, 0]
                )

    def validation(field_in, field_out, *, domain, origin):
        @gtscript.stencil(backend="numpy")
        def ref_stencil(
            field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]
        ):
            with computation(PARALLEL), interval(...):
                field_out = (  # noqa: F841  # Local name is assigned to but never used
                    field_in[-1, 0, 0] + field_in[0, 1, 0]
                )

        ref_stencil(field_in, field_out, domain=domain, origin=origin)


class TestRegionNoExtendLow(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[1:Iend, :]):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    -1, 0, 0
                ]

    def validation(field_in, field_out, *, domain, origin):
        @gtscript.stencil(backend="numpy")
        def ref_stencil(
            field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    -1, 0, 0
                ]

        domain = (domain[0] - 1, domain[1], domain[2])
        origin = {k: (orig[0] + 1, orig[1], orig[2]) for k, orig in origin.items()}
        ref_stencil(field_in, field_out, domain=domain, origin=origin)


class TestRegionNoExtendHigh(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[:, 0 : Jend - 1]):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    0, 1, 0
                ]

    def validation(field_in, field_out, *, domain, origin):
        @gtscript.stencil(backend="numpy")
        def ref_stencil(
            field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]
        ):
            with computation(PARALLEL), interval(...):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    0, 1, 0
                ]

        domain = (domain[0], domain[1] - 1, domain[2])
        origin = {k: (orig[0], orig[1], orig[2]) for k, orig in origin.items()}
        ref_stencil(field_in, field_out, domain=domain, origin=origin)

        # "j_end": AxisIndex(axis=J, index=-1, offset=0),


class TestRegionNoExtendHighSingleIdx(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[:, Jend - 2]):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    0, 1, 0
                ]

    def validation(field_in, field_out, *, domain, origin):
        # @gtscript.stencil(backend="numpy")
        # def ref_stencil(
        #     field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]
        # ):
        #     with computation(PARALLEL), interval(...):
        #         field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
        #             0, 1, 0
        #         ]

        field_out[:, -2, :] = field_in[:, -1, :]
        # domain = (domain[0], domain[1]-1, domain[2])
        # origin = {k: (orig[0], orig[1], orig[2]) for k, orig in origin.items()}
        # ref_stencil(field_in, field_out, domain=domain, origin=origin)


class TestRegionReadInsideLow(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[:, Jend - 1]):
                field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
                    0, -1, 0
                ]

    def validation(field_in, field_out, *, domain, origin):
        # @gtscript.stencil(backend="numpy")
        # def ref_stencil(
        #     field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]
        # ):
        #     with computation(PARALLEL), interval(...):
        #         field_out = field_in[  # noqa: F841  # Local name is assigned to but never used
        #             0, 1, 0
        #         ]

        field_out[:, -1, :] = field_in[:, -2, :]
        # domain = (domain[0], domain[1]-1, domain[2])
        # origin = {k: (orig[0], orig[1], orig[2]) for k, orig in origin.items()}
        # ref_stencil(field_in, field_out, domain=domain, origin=origin)


class TestRegionExtendHighRange(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "Iend": gt_testing.global_name(singleton=Iend),
        "Jend": gt_testing.global_name(singleton=Jend),
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            from __externals__ import Iend, Jend

            with horizontal(region[: Iend - 2, :]):
                field_out = field_in[1, 0, 0]

    def validation(field_in, field_out, *, domain, origin):
        field_out[:-2, :, :] = field_in[1:-1, :, :]


class TestRegionReadInsideHigh(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(4, 4), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            with horizontal(region[:, :3]):
                field_out = field_in[0, 1, 0]

    def validation(field_in, field_out, *, domain, origin):
        field_out[:, :3, :] = field_in[:, 1:4, :]


class TestRegionDoubleRegion(gt_testing.StencilTestSuite):
    dtypes = {
        "field_in": np.float64,
        "field_out": np.float64,
    }
    domain_range = [(10, 10), (4, 4), (4, 4)]
    backends = INTERNAL_BACKENDS
    symbols = {
        "field_in": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
        "field_out": gt_testing.field(
            in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
        ),
    }

    def definition(field_in, field_out):
        with computation(PARALLEL), interval(...):
            with horizontal(region[3, :], region[Iend - 3, :]):
                field_out = field_in[-1, 0, 0]

    def validation(field_in, field_out, *, domain, origin):
        field_out[3, :, :] = field_in[2, :, :]
        field_out[-3, :, :] = field_in[-4, :, :]


# comact 4-pt cubic interpolation
c1 = 2.0 / 3.0
c2 = -1.0 / 6.0
d1 = 0.375
d2 = -1.0 / 24.0
# PPM volume mean form
b1 = 7.0 / 12.0
b2 = -1.0 / 12.0
# 4-pt Lagrange interpolation
a1 = 9.0 / 16.0
a2 = -1.0 / 16.0

FloatField = gtscript.Field[np.float64]
from gt4py.gtscript import AxisIndex, I, J, K


FloatFieldIJ = gtscript.Field[gtscript.IJ, np.float64]

N = 12
NK = 80
import gt4py


@pytest.mark.parametrize("cand_backend", INTERNAL_BACKENDS)
def test_Region_divergence_corner(cand_backend):
    # def a2b_interpolation(
    #     tmp_qout_edges: FloatField,
    #     qout: FloatField,
    #     qx: FloatField,
    #     qy: FloatField,
    # ):
    #     from __externals__ import i_end, i_start, j_end, j_start
    #
    #     with computation(PARALLEL), interval(...):
    #         with horizontal(region[:, j_end]):
    #             qout = tmp_qout_edges[0, 1, 0]  # (qx[0, -1, 0] + qx) +
    def divergence_corner(
        u: FloatField,
        v: FloatField,
        ua: FloatField,
        va: FloatField,
        dxc: FloatFieldIJ,
        dyc: FloatFieldIJ,
        sin_sg1: FloatFieldIJ,
        sin_sg2: FloatFieldIJ,
        sin_sg3: FloatFieldIJ,
        sin_sg4: FloatFieldIJ,
        cos_sg1: FloatFieldIJ,
        cos_sg2: FloatFieldIJ,
        cos_sg3: FloatFieldIJ,
        cos_sg4: FloatFieldIJ,
        rarea_c: FloatFieldIJ,
        divg_d: FloatField,
    ):
        """Calculate divg on d-grid.
        Args:
            u: x-velocity (input)
            v: y-velocity (input)
            ua: x-velocity on a (input)
            va: y-velocity on a (input)
            dxc: grid spacing in x-direction (input)
            dyc: grid spacing in y-direction (input)
            sin_sg1: grid sin(sg1) (input)
            sin_sg2: grid sin(sg2) (input)
            sin_sg3: grid sin(sg3) (input)
            sin_sg4: grid sin(sg4) (input)
            cos_sg1: grid cos(sg1) (input)
            cos_sg2: grid cos(sg2) (input)
            cos_sg3: grid cos(sg3) (input)
            cos_sg4: grid cos(sg4) (input)
            rarea_c: inverse cell areas on c-grid (input)
            divg_d: divergence on d-grid (output)
        """
        from __externals__ import i_end, i_start, j_end, j_start

        with computation(PARALLEL), interval(...):

            uf = va[-1, 0, 0]
            vf = 0.0
            divg_d = vf[0, -1,  0] - vf + uf[-1, 0, 0] - uf

    # cand_backend = "gtc:dace"
    # cand_backend = "gtc:gt:cpu_ifirst"
    # cand_backend = "numpy"

    externals = {
        "i_start": AxisIndex(axis=I, index=0, offset=0),
        "local_is": AxisIndex(axis=I, index=0, offset=0),
        "i_end": AxisIndex(axis=I, index=-1, offset=-1),
        "local_ie": AxisIndex(axis=I, index=-1, offset=-1),
        "j_start": AxisIndex(axis=J, index=0, offset=0),
        "local_js": AxisIndex(axis=J, index=0, offset=0),
        "j_end": AxisIndex(axis=J, index=-1, offset=-1),
        "local_je": AxisIndex(axis=J, index=-1, offset=-1),
    }

    ref_stencil = gtscript.stencil(
        definition=divergence_corner, backend="numpy", rebuild=True, externals=externals
    )
    dace_stencil = gtscript.stencil(
        definition=divergence_corner, backend=cand_backend, rebuild=True, externals=externals
    )

    data1 = np.random.randn(N + 7, N + 7, NK)
    data2 = np.random.randn(N + 7, N + 7, NK)
    data3 = np.random.randn(N + 7, N + 7, NK)
    data4 = np.random.randn(N + 7, N + 7, NK)
    data5 = np.random.randn(N + 7, N + 7)
    data6 = np.random.randn(N + 7, N + 7)
    data7 = np.random.randn(N + 7, N + 7)
    data8 = np.random.randn(N + 7, N + 7)
    data9 = np.random.randn(N + 7, N + 7)
    data10 = np.random.randn(N + 7, N + 7)
    data11 = np.random.randn(N + 7, N + 7)
    data12 = np.random.randn(N + 7, N + 7)
    data13 = np.random.randn(N + 7, N + 7)
    data14 = np.random.randn(N + 7, N + 7)
    data15 = np.random.randn(N + 7, N + 7)
    u = gt4py.storage.from_array(
        data=data1,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    v = gt4py.storage.from_array(
        data=data2,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    ua = gt4py.storage.from_array(
        data=data3,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    va = gt4py.storage.from_array(
        data=data4,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    dxc = gt4py.storage.from_array(
        data=data5,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    dyc = gt4py.storage.from_array(
        data=data6,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    sin_sg1 = gt4py.storage.from_array(
        data=data7,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    sin_sg2 = gt4py.storage.from_array(
        data=data8,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    sin_sg3 = gt4py.storage.from_array(
        data=data9,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    sin_sg4 = gt4py.storage.from_array(
        data=data10,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    cos_sg1 = gt4py.storage.from_array(
        data=data11,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    cos_sg2 = gt4py.storage.from_array(
        data=data12,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    cos_sg3 = gt4py.storage.from_array(
        data=data13,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    cos_sg4 = gt4py.storage.from_array(
        data=data14,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    rarea_c = gt4py.storage.from_array(
        data=data15,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    # tmp_qout_edges = gt4py.storage.from_array(
    #     data=data1,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qx = gt4py.storage.from_array(
    #     data=data2,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qy = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # # qx_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    # # qy_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    divg_d_ref = gt4py.storage.zeros(
        "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    # qx_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    # qy_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    divg_d_cand = gt4py.storage.zeros(
        cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    ref_stencil(
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
        divg_d_ref,
        domain=(N, N, NK - 3),
    )
    dace_stencil(
        u,
        v,
        ua,
        va,
        dxc,
        dyc,
        sin_sg1,
        sin_sg2,
        sin_sg3,
        sin_sg4,
        cos_sg1,
        cos_sg2,
        cos_sg3,
        cos_sg4,
        rarea_c,
        divg_d_cand,
        domain=(N, N, NK - 3),
    )

    np.testing.assert_allclose(divg_d_cand, divg_d_ref)
    # np.testing.assert_allclose(qx_cand, qx_ref)
    # np.testing.assert_allclose(qy_cand, qy_ref)


# class TestRegionExtendHighAcutallyNotHaha(gt_testing.StencilTestSuite):
#     dtypes = {
#         "field_in": np.float64,
#         "field_out": np.float64,
#     }
#     domain_range = [(4, 4), (4, 4), (4, 4)]
#     backends = INTERNAL_BACKENDS
#     symbols = {
#         "field_in": gt_testing.field(
#             in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
#         ),
#         "field_out": gt_testing.field(
#             in_range=(0.1, 10), axes="IJK", boundary=[(0, 0), (0, 0), (0, 0)]
#         ),
#     }
#
#     def definition(field_in, field_out):
#         with computation(PARALLEL), interval(...):
#             field_out = field_in[0, -1, 0]
#
#     def validation(field_in, field_out, *, domain, origin):
#         # field_out[:, :-1, :] = field_in[:, 1:, :]
#         field_out[:, 1:, :] = field_in[:, :-1, :]


@pytest.mark.parametrize("cand_backend", INTERNAL_BACKENDS)
def test_fill_corners(cand_backend):
    @gtscript.function
    def fill_corners_2cells_mult_x(
        q: FloatField,
        q_corner: FloatField,
        sw_mult: float,
        se_mult: float,
        nw_mult: float,
        ne_mult: float,
    ):
        """
        Fills cell quantity q using corners from q_corner and multipliers in x-dir.
        """
        from __externals__ import i_end, i_start, j_end, j_start

        # Southwest
        with horizontal(region[i_start - 1, j_start - 1]):
            q = sw_mult * q_corner[0, 1, 0]
        with horizontal(region[i_start - 2, j_start - 1]):
            q = sw_mult * q_corner[1, 2, 0]

        # Southeast
        with horizontal(region[i_end + 1, j_start - 1]):
            q = se_mult * q_corner[0, 1, 0]
        with horizontal(region[i_end + 2, j_start - 1]):
            q = se_mult * q_corner[-1, 2, 0]

        # Northwest
        with horizontal(region[i_start - 1, j_end + 1]):
            q = nw_mult * q_corner[0, -1, 0]
        with horizontal(region[i_start - 2, j_end + 1]):
            q = nw_mult * q_corner[1, -2, 0]

        # Northeast
        with horizontal(region[i_end + 1, j_end + 1]):
            q = ne_mult * q_corner[0, -1, 0]
        with horizontal(region[i_end + 2, j_end + 1]):
            q = ne_mult * q_corner[-1, -2, 0]

        return q

    @gtscript.function
    def fill_corners_3cells_mult_x(
        q: FloatField,
        q_corner: FloatField,
        sw_mult: float,
        se_mult: float,
        nw_mult: float,
        ne_mult: float,
    ):
        """
        Fills cell quantity q using corners from q_corner and multipliers in x-dir.
        """
        from __externals__ import i_end, i_start, j_end, j_start

        q = fill_corners_2cells_mult_x(q, q_corner, sw_mult, se_mult, nw_mult, ne_mult)

        # Southwest
        with horizontal(region[i_start - 3, j_start - 1]):
            q = sw_mult * q_corner[2, 3, 0]

        # Southeast
        with horizontal(region[i_end + 3, j_start - 1]):
            q = se_mult * q_corner[-2, 3, 0]
            q = nw_mult * q_corner[2, -3, 0]

        # Northwest
        with horizontal(region[i_start - 3, j_end + 1]):
            q = nw_mult * q_corner[2, -3, 0]

        # Northeast
        with horizontal(region[i_end + 3, j_end + 1]):
            q = ne_mult * q_corner[-2, -3, 0]

        return q

    def fill_corners_x(utmp: FloatField, vtmp: FloatField, ua: FloatField, va: FloatField):
        with computation(PARALLEL), interval(...):
            utmp = fill_corners_3cells_mult_x(
                utmp, vtmp, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1
            )
            # ua = fill_corners_2cells_mult_x(ua, va, sw_mult=-1, se_mult=1, ne_mult=-1, nw_mult=1)

    # cand_backend = "gtc:dace"
    # cand_backend = "gtc:dace"
    # cand_backend = "numpy"

    externals = {
        "i_start": AxisIndex(axis=I, index=0, offset=3),
        "local_is": AxisIndex(axis=I, index=0, offset=3),
        "i_end": AxisIndex(axis=I, index=-1, offset=-3),
        "local_ie": AxisIndex(axis=I, index=-1, offset=-3),
        "j_start": AxisIndex(axis=J, index=0, offset=3),
        "local_js": AxisIndex(axis=J, index=0, offset=3),
        "j_end": AxisIndex(axis=J, index=-1, offset=-3),
        "local_je": AxisIndex(axis=J, index=-1, offset=-3),
    }

    ref_stencil = gtscript.stencil(
        definition=fill_corners_x, backend="numpy", rebuild=True, externals=externals
    )
    dace_stencil = gtscript.stencil(
        definition=fill_corners_x, backend=cand_backend, rebuild=True, externals=externals
    )

    data1 = np.random.randn(N + 7, N + 7, NK)
    data2 = np.random.randn(N + 7, N + 7, NK)
    data3 = np.random.randn(N + 7, N + 7, NK)
    data4 = np.random.randn(N + 7, N + 7, NK)
    data5 = np.random.randn(N + 7, N + 7)
    data6 = np.random.randn(N + 7, N + 7)
    data7 = np.random.randn(N + 7, N + 7)
    data8 = np.random.randn(N + 7, N + 7)
    data9 = np.random.randn(N + 7, N + 7)
    data10 = np.random.randn(N + 7, N + 7)
    data11 = np.random.randn(N + 7, N + 7)
    data12 = np.random.randn(N + 7, N + 7)
    data13 = np.random.randn(N + 7, N + 7)
    data14 = np.random.randn(N + 7, N + 7)
    data15 = np.random.randn(N + 7, N + 7)
    vtmp = gt4py.storage.from_array(
        data=data1,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    va = gt4py.storage.from_array(
        data=data2,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    # ua = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # va = gt4py.storage.from_array(
    #     data=data4,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # dxc = gt4py.storage.from_array(
    #     data=data5,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # dyc = gt4py.storage.from_array(
    #     data=data6,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg1 = gt4py.storage.from_array(
    #     data=data7,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg2 = gt4py.storage.from_array(
    #     data=data8,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg3 = gt4py.storage.from_array(
    #     data=data9,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg4 = gt4py.storage.from_array(
    #     data=data10,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg1 = gt4py.storage.from_array(
    #     data=data11,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg2 = gt4py.storage.from_array(
    #     data=data12,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg3 = gt4py.storage.from_array(
    #     data=data13,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg4 = gt4py.storage.from_array(
    #     data=data14,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # rarea_c = gt4py.storage.from_array(
    #     data=data15,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # tmp_qout_edges = gt4py.storage.from_array(
    #     data=data1,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qx = gt4py.storage.from_array(
    #     data=data2,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qy = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # # qx_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    # # qy_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    utmp_ref = gt4py.storage.zeros(
        "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )
    ua_ref = gt4py.storage.zeros(
        "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    # qx_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    # qy_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    utmp_cand = gt4py.storage.zeros(
        cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )
    ua_cand = gt4py.storage.zeros(
        cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    ref_stencil(
        vtmp=vtmp,
        va=va,
        utmp=utmp_ref,
        ua=ua_ref,
        domain=(N, N, NK - 3),
    )
    dace_stencil(
        vtmp=vtmp,
        va=va,
        utmp=utmp_cand,
        ua=ua_cand,
        domain=(N, N, NK - 3),
    )

    np.testing.assert_allclose(utmp_cand, utmp_ref)
    np.testing.assert_allclose(ua_cand, ua_ref)
    # np.testing.assert_allclose(qx_cand, qx_ref)
    # np.testing.assert_allclose(qy_cand, qy_ref)


@pytest.mark.parametrize("cand_backend", INTERNAL_BACKENDS)
def test_Region_uc_contra_y_edge(cand_backend):
    def uc_contra_y_edge(
        uc: FloatField,
        sin_sg1: FloatFieldIJ,
        sin_sg3: FloatFieldIJ,
        uc_contra: FloatField,
    ):
        from __externals__ import i_end, i_start

        with computation(PARALLEL), interval(...):
            with horizontal(region[i_start, :], region[i_end + 1, :]):
                uc_contra = uc / sin_sg3[-1, 0]

    # cand_backend = "gtc:dace"
    # cand_backend = "gtc:gt:cpu_ifirst"
    # cand_backend = "numpy"

    externals = {
        "i_start": AxisIndex(axis=I, index=0, offset=3),
        "local_is": AxisIndex(axis=I, index=0, offset=3),
        "i_end": AxisIndex(axis=I, index=-1, offset=-3),
        "local_ie": AxisIndex(axis=I, index=-1, offset=-3),
        "j_start": AxisIndex(axis=J, index=0, offset=3),
        "local_js": AxisIndex(axis=J, index=0, offset=3),
        "j_end": AxisIndex(axis=J, index=-1, offset=-3),
        "local_je": AxisIndex(axis=J, index=-1, offset=-3),
    }

    ref_stencil = gtscript.stencil(
        definition=uc_contra_y_edge, backend="numpy", rebuild=True, externals=externals
    )
    dace_stencil = gtscript.stencil(
        definition=uc_contra_y_edge, backend=cand_backend, rebuild=True, externals=externals
    )

    data1 = np.random.randn(N + 7, N + 7, NK)
    # data2 = np.random.randn(N + 7, N + 7, NK)
    # data3 = np.random.randn(N + 7, N + 7, NK)
    # data4 = np.random.randn(N + 7, N + 7, NK)
    # data5 = np.random.randn(N + 7, N + 7)
    # data6 = np.random.randn(N + 7, N + 7)
    data7 = np.random.randn(N + 7, N + 7)
    # data8 = np.random.randn(N + 7, N + 7)
    data9 = np.random.randn(N + 7, N + 7)
    # data10 = np.random.randn(N + 7, N + 7)
    # data11 = np.random.randn(N + 7, N + 7)
    # data12 = np.random.randn(N + 7, N + 7)
    # data13 = np.random.randn(N + 7, N + 7)
    # data14 = np.random.randn(N + 7, N + 7)
    # data15 = np.random.randn(N + 7, N + 7)
    uc = gt4py.storage.from_array(
        data=data1,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    # va = gt4py.storage.from_array(
    #     data=data2,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # ua = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # va = gt4py.storage.from_array(
    #     data=data4,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # dxc = gt4py.storage.from_array(
    #     data=data5,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # dyc = gt4py.storage.from_array(
    #     data=data6,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    sin_sg1 = gt4py.storage.from_array(
        data=data7,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    # sin_sg2 = gt4py.storage.from_array(
    #     data=data8,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    sin_sg3 = gt4py.storage.from_array(
        data=data9,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7),
        mask=[True, True, False],
        dtype=np.float64,
    )
    # sin_sg4 = gt4py.storage.from_array(
    #     data=data10,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg1 = gt4py.storage.from_array(
    #     data=data11,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg2 = gt4py.storage.from_array(
    #     data=data12,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg3 = gt4py.storage.from_array(
    #     data=data13,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg4 = gt4py.storage.from_array(
    #     data=data14,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # rarea_c = gt4py.storage.from_array(
    #     data=data15,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # tmp_qout_edges = gt4py.storage.from_array(
    #     data=data1,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qx = gt4py.storage.from_array(
    #     data=data2,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qy = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # # qx_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    # # qy_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    uc_contra_ref = gt4py.storage.zeros(
        "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )
    # ua_ref = gt4py.storage.zeros(
    #     "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )

    # qx_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    # qy_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    uc_contra_cand = gt4py.storage.zeros(
        cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )
    # ua_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )

    ref_stencil(
        uc=uc,
        sin_sg1=sin_sg1,
        sin_sg3=sin_sg3,
        uc_contra=uc_contra_ref,
        domain=(N, N, NK - 3),
    )
    dace_stencil(
        uc=uc,
        sin_sg1=sin_sg1,
        sin_sg3=sin_sg3,
        uc_contra=uc_contra_cand,
        domain=(N, N, NK - 3),
    )

    np.testing.assert_allclose(uc_contra_cand, uc_contra_ref)
    # np.testing.assert_allclose(ua_cand, ua_ref)
    # np.testing.assert_allclose(qx_cand, qx_ref)
    # np.testing.assert_allclose(qy_cand, qy_ref)


@pytest.mark.parametrize("cand_backend", INTERNAL_BACKENDS)
def test_avg_box(cand_backend):
    @gtscript.stencil(
        backend=cand_backend,
    )
    def avg_box(u: FloatField, v: FloatField, utmp: FloatField, vtmp: FloatField):
        with computation(PARALLEL), interval(...):
            with horizontal(
                region[:, :],
            ):
                utmp = 0.5 * (u + u[0, 1, 0])
                # vtmp = 0.5 * (v + v[1, 0, 0])


@pytest.mark.parametrize("cand_backend", INTERNAL_BACKENDS)
def test_Region_a2b_interpolation(cand_backend):
    def a2b_interpolation(
        tmp_qout_edges: FloatField,
        qout: FloatField,
        qx: FloatField,
        qy: FloatField,
    ):
        from __externals__ import i_end, i_start, j_end, j_start

        with computation(PARALLEL), interval(...):
            qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
            qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)
            # TODO(rheag) use a function with an offset when that works consistently
            with horizontal(region[:, j_start + 1]):
                qxx_upper = a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])
                qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (tmp_qout_edges[0, -1, 0] + qxx_upper)
            with horizontal(region[:, j_end]):
                qxx_lower = a2 * (qx[0, -3, 0] + qx) + a1 * (qx[0, -2, 0] + qx[0, -1, 0])
                qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (tmp_qout_edges[0, 1, 0] + qxx_lower)
            with horizontal(region[i_start + 1, :]):
                qyy_right = a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])
                qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (tmp_qout_edges[-1, 0, 0] + qyy_right)
            with horizontal(region[i_end, :]):
                qyy_left = a2 * (qy[-3, 0, 0] + qy) + a1 * (qy[-2, 0, 0] + qy[-1, 0, 0])
                qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (tmp_qout_edges[1, 0, 0] + qyy_left)
            qout = 0.5 * (qxx + qyy)

    # cand_backend = "gtc:dace"
    cand_backend = "gtc:gt:cpu_ifirst"
    # cand_backend = "numpy"

    externals = {
        "i_start": AxisIndex(axis=I, index=0, offset=-1),
        "local_is": AxisIndex(axis=I, index=0, offset=-1),
        "i_end": AxisIndex(axis=I, index=-1, offset=0),
        "local_ie": AxisIndex(axis=I, index=-1, offset=0),
        "j_start": AxisIndex(axis=J, index=0, offset=-1),
        "local_js": AxisIndex(axis=J, index=0, offset=-1),
        "j_end": AxisIndex(axis=J, index=-1, offset=0),
        "local_je": AxisIndex(axis=J, index=-1, offset=0),
    }

    ref_stencil = gtscript.stencil(
        definition=a2b_interpolation, backend="numpy", rebuild=True, externals=externals
    )
    dace_stencil = gtscript.stencil(
        definition=a2b_interpolation, backend=cand_backend, rebuild=True, externals=externals
    )

    data1 = np.random.randn(N + 7, N + 7, NK)
    data2 = np.random.randn(N + 7, N + 7, NK)
    data3 = np.random.randn(N + 7, N + 7, NK)
    data4 = np.random.randn(N + 7, N + 7, NK)
    data5 = np.random.randn(N + 7, N + 7)
    data6 = np.random.randn(N + 7, N + 7)
    data7 = np.random.randn(N + 7, N + 7)
    data8 = np.random.randn(N + 7, N + 7)
    data9 = np.random.randn(N + 7, N + 7)
    data10 = np.random.randn(N + 7, N + 7)
    data11 = np.random.randn(N + 7, N + 7)
    data12 = np.random.randn(N + 7, N + 7)
    data13 = np.random.randn(N + 7, N + 7)
    data14 = np.random.randn(N + 7, N + 7)
    data15 = np.random.randn(N + 7, N + 7)
    # tmp_qout_edges = gt4py.storage.from_array(
    #     data=data1,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qx = gt4py.storage.from_array(
    #     data=data2,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # qy = gt4py.storage.from_array(
    #     data=data3,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # va = gt4py.storage.from_array(
    #     data=data4,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7, NK),
    #     dtype=np.float64,
    # )
    # dxc = gt4py.storage.from_array(
    #     data=data5,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # dyc = gt4py.storage.from_array(
    #     data=data6,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg1 = gt4py.storage.from_array(
    #     data=data7,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg2 = gt4py.storage.from_array(
    #     data=data8,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg3 = gt4py.storage.from_array(
    #     data=data9,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # sin_sg4 = gt4py.storage.from_array(
    #     data=data10,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg1 = gt4py.storage.from_array(
    #     data=data11,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg2 = gt4py.storage.from_array(
    #     data=data12,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg3 = gt4py.storage.from_array(
    #     data=data13,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # cos_sg4 = gt4py.storage.from_array(
    #     data=data14,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    # rarea_c = gt4py.storage.from_array(
    #     data=data15,
    #     backend=cand_backend,
    #     default_origin=(3, 3, 3),
    #     shape=(N + 7, N + 7),
    #     mask=[True, True, False],
    #     dtype=np.float64,
    # )
    tmp_qout_edges = gt4py.storage.from_array(
        data=data1,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    qx = gt4py.storage.from_array(
        data=data2,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    qy = gt4py.storage.from_array(
        data=data3,
        backend=cand_backend,
        default_origin=(3, 3, 3),
        shape=(N + 7, N + 7, NK),
        dtype=np.float64,
    )
    # # qx_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    # # qy_ref = gt4py.storage.zeros(
    # #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # # )
    qout_ref = gt4py.storage.ones(
        "numpy", default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    # qx_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    # qy_cand = gt4py.storage.zeros(
    #     cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    # )
    qout_cand = gt4py.storage.ones(
        cand_backend, default_origin=(3, 3, 3), shape=(N + 7, N + 7, NK), dtype=np.float64
    )

    ref_stencil(
        tmp_qout_edges=tmp_qout_edges,
        qout=qout_ref,
        qx=qx,
        qy=qy,
        domain=(N, N, NK - 3),
    )
    dace_stencil(
        tmp_qout_edges=tmp_qout_edges,
        qout=qout_cand,
        qx=qx,
        qy=qy,
        domain=(N, N, NK - 3),
    )

    np.testing.assert_allclose(qout_cand, qout_ref)
    # np.testing.assert_allclose(qx_cand, qx_ref)
    # np.testing.assert_allclose(qy_cand, qy_ref)
