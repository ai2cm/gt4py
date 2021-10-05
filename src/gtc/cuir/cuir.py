# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from pydantic import validator

from eve import Str, SymbolName, SymbolTableTrait, field, utils
from gtc import common
from gtc.common import AxisBound, CartesianOffset, DataType, LocNode, LoopOrder


@utils.noninstantiable
class Expr(common.Expr):
    dtype: common.DataType


@utils.noninstantiable
class Stmt(common.Stmt):
    pass


class Literal(common.Literal, Expr):  # type: ignore
    pass


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class FieldAccess(common.FieldAccess, Expr):  # type: ignore
    in_horizontal_mask: bool = False


class IJCacheAccess(common.FieldAccess, Expr):
    ij_cache_is_different_from_field_access = True
    in_horizontal_mask: bool = False

    @validator("offset")
    def zero_k_offset(cls, v: CartesianOffset) -> CartesianOffset:
        if v.k != 0:
            raise ValueError("No k-offset allowed")
        return v

    @validator("data_index")
    def no_additional_dimensions(cls, v: List[int]) -> List[int]:
        if v:
            raise ValueError("IJ-cached higher-dimensional fields are not supported")
        return v


class KCacheAccess(common.FieldAccess, Expr):
    k_cache_is_different_from_field_access = True

    @validator("offset")
    def zero_ij_offset(cls, v: CartesianOffset) -> CartesianOffset:
        if not v.i == v.j == 0:
            raise ValueError("No ij-offset allowed")
        return v

    @validator("data_index")
    def no_additional_dimensions(cls, v: List[int]) -> List[int]:
        if v:
            raise ValueError("K-cached higher-dimensional fields are not supported")
        return v


class AssignStmt(
    common.AssignStmt[Union[ScalarAccess, FieldAccess, IJCacheAccess, KCacheAccess], Expr], Stmt
):
    _dtype_validation = common.assign_stmt_dtype_validation(strict=True)


class MaskStmt(Stmt):
    mask: Expr
    body: List[Stmt]


class While(common.While[Stmt, Expr], Stmt):
    pass


class For(Stmt):
    target_name: Str
    start: Expr
    end: Expr
    inc: int
    body: List[Stmt]


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class VariableOffset(common.VariableOffset):
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Decl(LocNode):
    name: SymbolName
    dtype: DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    dimensions: Tuple[bool, bool, bool]
    data_dims: Tuple[int, ...] = field(default_factory=tuple)


class ScalarDecl(Decl):
    pass


class LocalScalar(Decl):
    pass


class Temporary(Decl):
    pass


class IJExtent(common.IJExtent):
    pass


class KExtent(LocNode):
    k: Tuple[int, int]

    @classmethod
    def zero(cls) -> "KExtent":
        return cls(k=(0, 0))

    @classmethod
    def from_offset(cls, offset: CartesianOffset) -> "KExtent":
        k_offset = offset.to_dict()["k"]
        return cls(k=(k_offset, k_offset))

    def union(*extents: "KExtent") -> "KExtent":
        return KExtent(k=(min(e.k[0] for e in extents), max(e.k[1] for e in extents)))


class IJCacheDecl(Decl):
    extent: Optional[IJExtent]


class KCacheDecl(Decl):
    extent: Optional[KExtent]


class HorizontalExecution(LocNode):
    body: List[Stmt]
    declarations: List[LocalScalar]
    extent: Optional[IJExtent]


class VerticalLoopSection(LocNode):
    start: AxisBound
    end: AxisBound
    horizontal_executions: List[HorizontalExecution]


class VerticalLoop(LocNode):
    loop_order: LoopOrder
    sections: List[VerticalLoopSection]
    ij_caches: List[IJCacheDecl]
    k_caches: List[KCacheDecl]
    has_horizontal_masks: bool = False


class Kernel(LocNode):
    vertical_loops: List[VerticalLoop]

    @validator("vertical_loops")
    def check_loops(cls, v: List[VerticalLoop]) -> List[VerticalLoop]:
        if not v:
            raise ValueError("At least one loop required")
        parallel = [loop.loop_order == LoopOrder.PARALLEL for loop in v]
        if any(parallel) and not all(parallel):
            raise ValueError("Mixed k-parallelism in kernel")
        return v


@dataclass
class DependencyGraph:
    row_ind: List[int]
    col_ind: List[int]
    val_ind: Optional[List[int]] = None


class Program(LocNode, SymbolTableTrait):
    name: Str
    params: List[Decl]
    temporaries: List[Temporary]
    kernels: List[Kernel]
    # node i dependent on j: dependency[i, j] != 0, stored in CRS format
    # DependencyGraph(row_ind, col_ind), dependency[i, j] = j in col_ind[i: i + 1]
    # None means uninitialized, use serial semantics as default
    dependency: Optional[DependencyGraph]
