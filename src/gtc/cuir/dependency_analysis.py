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

from itertools import accumulate, chain
from typing import List, Optional, Set

from eve import NodeTranslator

from . import cuir


class DependencyAnalysis(NodeTranslator):
    """
    Dependency analysis for CUIR kernels, store dependency array in Program node
    """

    # TODO: ignore false dependency introduced by aliasing temporary variables
    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        dependency_arr_full: List[List[int]] = [
            [] for _ in range(len(node.kernels))
        ]  # with full transitive relation
        dependency_arr: List[List[int]] = [
            [] for _ in range(len(node.kernels))
        ]  # without transitive relation
        writes = []
        reads = []
        for i in range(len(node.kernels)):
            kernel = node.kernels[i]
            writes.append(
                kernel.iter_tree()
                .if_isinstance(cuir.AssignStmt)
                .getattr("left")
                .if_isinstance(cuir.FieldAccess)
                .getattr("name")
                .to_set()
            )
            reads.append(
                kernel.iter_tree().if_isinstance(cuir.FieldAccess).getattr("name").to_set()
                - writes[i]
            )
            for j in reversed(range(0, i)):
                dep_flag = False
                # R -> W, W -> W
                if writes[i].intersection(writes[j].union(reads[j])):
                    dep_flag = True
                # W -> R
                if reads[i].intersection(writes[j]):
                    dep_flag = True
                if dep_flag:
                    transitive_flag = False
                    for k in dependency_arr_full[i]:
                        # j -> k -> i implies j -> i
                        if j in dependency_arr_full[k]:
                            transitive_flag = True
                            break
                    if not transitive_flag:
                        dependency_arr[i].append(j)
                    dependency_arr_full[i].append(j)
        row_ind = list(accumulate([len(arr) for arr in dependency_arr], initial=0))
        col_ind = list(chain.from_iterable(dependency_arr))
        return cuir.Program(
            name=node.name,
            params=node.params,
            temporaries=node.temporaries,
            kernels=node.kernels,
            dependency=cuir.DependencyGraph(row_ind=row_ind, col_ind=col_ind),
        )


class FuseKernels(NodeTranslator):
    """
    Fuse Kernels based on dependency analysis
    ==> K1 -> K2 -> K3 -> K4 ==>
    Fused to
    ==> Fused K ==>
    """

    def mark_fusable_kernels(self, graph: cuir.DependencyGraph):
        row_ind = graph.row_ind
        col_ind = graph.col_ind
        num_kernels = len(row_ind) - 1
        degree_in: List[int] = [
            row_ind[i + 1] - row_ind[i] for i in range(num_kernels)
        ]  # i -> degree_in_i
        degree_out: List[int] = [col_ind.count(i) for i in range(num_kernels)]
        unfused_kernels: Set[int] = set(range(num_kernels))

        def prev_link(i) -> Optional[int]:
            if degree_in[i] == 1:
                prev = col_ind[row_ind[i]]
                if degree_out[prev] == 1:
                    return prev
            return None

        def find_link_end(unfused_kernels):
            ret = None
            for i in sorted(unfused_kernels, reverse=True):
                if degree_in[i] == 1:
                    prev = prev_link(i)
                    if prev in unfused_kernels:
                        ret = i
                        break
            return ret

        fused_kernels: List[Set[int]] = []
        link_end = find_link_end(unfused_kernels)
        while link_end:
            fused_kernels_i: Set[int] = {link_end}
            prev = prev_link(link_end)
            while prev in unfused_kernels:
                fused_kernels_i.add(prev)
                prev = prev_link(prev)
            unfused_kernels = unfused_kernels - fused_kernels_i
            fused_kernels.append(fused_kernels_i)
            link_end = find_link_end(unfused_kernels)
        return fused_kernels

    @staticmethod
    def is_parallel(kernel: cuir.Kernel) -> bool:
        parallel = [loop.loop_order == cuir.LoopOrder.PARALLEL for loop in kernel.vertical_loops]
        assert all(parallel) or not any(parallel), "Mixed k-parallelism in kernel"
        return any(parallel)

    def visit_VerticalLoop(
        self, node: cuir.VerticalLoop, loop_order: Optional[cuir.LoopOrder]
    ) -> cuir.VerticalLoop:
        pass

    def visit_Program(self, node: cuir.Program) -> cuir.Program:
        fusable_kernels = self.mark_fusable_kernels(node.dependency)
        fused_kernel_inds: List[Optional[int]] = [None] * len(fusable_kernels)
        return node
