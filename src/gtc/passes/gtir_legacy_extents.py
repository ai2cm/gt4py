from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from eve import NodeVisitor
from eve.utils import XIterable
from gt4py.definitions import Extent
from gtc import gtir
from gtc.passes import utils


def _iter_field_names(node: Union[gtir.Stencil, gtir.ParAssignStmt]) -> XIterable[gtir.FieldAccess]:
    return node.iter_tree().if_isinstance(gtir.FieldDecl).getattr("name").unique()


def _iter_assigns(node: gtir.Stencil) -> XIterable[gtir.ParAssignStmt]:
    return node.iter_tree().if_isinstance(gtir.ParAssignStmt)


def _ext_from_off(offset: gtir.CartesianOffset) -> Extent:
    return Extent(
        ((min(offset.i, 0), max(offset.i, 0)), (min(offset.j, 0), max(offset.j, 0)), (0, 0))
    )


FIELD_EXT_T = Dict[str, Extent]


class LegacyExtentsVisitor(NodeVisitor):
    def __init__(self, allow_negative=False):
        self._allow_negative = allow_negative
        super().__init__()

    @dataclass
    class AssignContext:
        left_extent: Extent
        assign_extents: Dict[str, Extent] = field(default_factory=dict)

    @dataclass
    class StencilContext:
        assign_conditions: Dict[int, List[gtir.FieldAccess]] = field(default_factory=dict)
        assign_masks: Dict[int, List[gtir.HorizontalMask]] = field(default_factory=dict)

    def visit_Stencil(self, node: gtir.Stencil, **kwargs: Any) -> FIELD_EXT_T:
        field_extents: FIELD_EXT_T = dict()
        ctx = self.StencilContext()
        for field_if in node.iter_tree().if_isinstance(gtir.FieldIfStmt):
            self.visit(field_if, ctx=ctx)
        for region in node.iter_tree().if_isinstance(gtir.HorizontalRegion):
            self.visit(region, ctx=ctx)
        for assign in reversed(_iter_assigns(node).to_list()):
            self.visit(assign, ctx=ctx, field_extents=field_extents)
        for name in _iter_field_names(node):
            if name not in field_extents:
                field_extents[name] = Extent.zeros()
        return field_extents

    def visit_HorizontalRegion(
        self,
        node: gtir.HorizontalRegion,
        *,
        ctx: StencilContext,
        **kwargs: Any,
    ) -> None:
        ctx.assign_masks.update({id(assign): node.mask for assign in _iter_assigns(node).to_list()})

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: StencilContext,
        field_extents: Dict[str, Extent],
        **kwargs: Any,
    ) -> None:
        horizontal_mask = ctx.assign_masks.get(id(node), None)
        left_extent = field_extents.setdefault(node.left.name, Extent.zeros())
        if horizontal_mask:
            dist_from_edge = utils.compute_extent_difference(left_extent, horizontal_mask)
            if dist_from_edge is None:
                return
        else:
            dist_from_edge = Extent.zeros()
        pa_ctx = self.AssignContext(left_extent=left_extent - dist_from_edge)
        self.visit(
            ctx.assign_conditions.get(id(node), []),
            field_extents=field_extents,
            pa_ctx=pa_ctx,
            **kwargs,
        )
        self.visit(node.right, field_extents=field_extents, pa_ctx=pa_ctx, **kwargs)
        for key, value in pa_ctx.assign_extents.items():
            field_extents[key] |= value

    def visit_FieldIfStmt(
        self, node: gtir.FieldIfStmt, *, ctx: StencilContext, **kwargs: Any
    ) -> None:
        for assign_id in node.iter_tree().if_isinstance(gtir.ParAssignStmt).map(id):
            ctx.assign_conditions.setdefault(assign_id, []).extend(
                node.cond.iter_tree().if_isinstance(gtir.FieldAccess).to_list()
            )

    def visit_FieldAccess(
        self,
        node: gtir.FieldAccess,
        *,
        field_extents: Dict[str, Extent],
        pa_ctx: AssignContext,
        **kwargs: Any,
    ) -> None:
        if self._allow_negative:
            default_extent = Extent.from_offset((node.offset.i, node.offset.j, 0))
        else:
            default_extent = Extent.zeros()

        pa_ctx.assign_extents.setdefault(
            node.name, field_extents.setdefault(node.name, default_extent)
        )
        if self._allow_negative:
            pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + Extent.from_offset(
                (node.offset.i, node.offset.j, 0)
            )
        else:
            pa_ctx.assign_extents[node.name] |= pa_ctx.left_extent + _ext_from_off(node.offset)


def compute_legacy_extents(node: gtir.Stencil, allow_negative=False) -> FIELD_EXT_T:
    return LegacyExtentsVisitor(allow_negative).visit(node)
