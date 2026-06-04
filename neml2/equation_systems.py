# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Python-native equation-system assembly for implicit updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import TYPE_CHECKING, Any, overload

import torch
from torch import nn

from .chain_rule import ChainRuleDict
from .factory import register_native
from .model import Model
from .schema import HitSchema, dependency, option
from .types import TensorWrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    import nmhit

    from .factory import _NativeInputFile


def _storage_size(type_cls: type[TensorWrapper]) -> int:
    return prod(type_cls.BASE_SHAPE) if type_cls.BASE_SHAPE else 1


def _batch_shape(tensor: torch.Tensor, type_cls: type[TensorWrapper]) -> torch.Size:
    if type_cls.BASE_NDIM == 0:
        return tensor.shape
    return tensor.shape[: -type_cls.BASE_NDIM]


def _flatten_base(tensor: torch.Tensor, type_cls: type[TensorWrapper]) -> torch.Tensor:
    if type_cls.BASE_NDIM == 0:
        return tensor.unsqueeze(-1)
    return tensor.reshape(*tensor.shape[: -type_cls.BASE_NDIM], _storage_size(type_cls))


def _unflatten_base(tensor: torch.Tensor, type_cls: type[TensorWrapper]) -> torch.Tensor:
    if type_cls.BASE_NDIM == 0:
        return tensor.squeeze(-1)
    return tensor.reshape(*tensor.shape[:-1], *type_cls.BASE_SHAPE)


def _flatten_sub_batch_and_base(
    tensor: torch.Tensor,
    type_cls: type[TensorWrapper],
    sub_batch_shape: tuple[int, ...] | torch.Size,
    dyn_ndim: int,
) -> torch.Tensor:
    """Collapse ``(*dyn, *sub_batch, *base)`` → ``(*dyn, prod(sub_batch) * base_size)``.

    Used by ``AssembledVector.from_dict`` and ``_assemble_matrix`` to flatten
    per-variable per-(crystal × base) DOFs into a single trailing dim so the
    assembled vector/matrix has uniform shape per group regardless of the
    variable's sub-batch ndim.
    """
    base_size = _storage_size(type_cls)
    sb_total = 1
    for s in sub_batch_shape:
        sb_total *= int(s)
    dyn_shape = tensor.shape[:dyn_ndim]
    return tensor.reshape(*dyn_shape, sb_total * base_size)


def _unflatten_sub_batch_and_base(
    tensor: torch.Tensor,
    type_cls: type[TensorWrapper],
    sub_batch_shape: tuple[int, ...] | torch.Size,
    dyn_ndim: int,
) -> torch.Tensor:
    """Inverse of :func:`_flatten_sub_batch_and_base`."""
    dyn_shape = tensor.shape[:dyn_ndim]
    # Pass as a single tuple so reshape() doesn't see zero positional args when
    # both dyn_shape and sub_batch_shape are empty (scalar unbatched unknowns).
    if type_cls.BASE_NDIM == 0:
        return tensor.reshape((*dyn_shape, *sub_batch_shape))
    return tensor.reshape((*dyn_shape, *sub_batch_shape, *type_cls.BASE_SHAPE))


def _expanded_identity_seed(
    type_cls: type[TensorWrapper],
    sub_batch_shape: tuple[int, ...] | torch.Size,
    dyn_shape: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorWrapper:
    """Per-variable identity seed expanded for sub-batch independence.

    tangents flow as ordinary typed wrappers with the
    seed-direction axis $K$ as the **leftmost** batch dim. This returns a
    ``type_cls`` wrapper whose ``data`` has shape
    ``(K, *dyn_shape, *sub_batch_shape, *BASE_SHAPE)`` (where
    $K = prod(sub_batch_shape) * base_size$) and
    ``sub_batch_ndim = len(sub_batch_shape)``.

    The K seed directions are block-diagonal in the (per-sub-batch-site ×
    per-base-component) sense — direction ``c × base_size + j`` is the unit
    perturbation of base index $j$ at sub-batch site $c$ (linear index
    across all sub-batch axes), zero elsewhere. Concretely we build the flat
    ``eye(K)`` and unflatten its column axis into ``(*sub_batch_shape,
    *BASE_SHAPE)``, then broadcast the dynamic batch in the middle.

    For variables with no sub-batch (``sub_batch_shape == ()``) this collapses
    to the leading-K ``eye(base_size)`` broadcast — the only placement where the
    tangent broadcasts cleanly against a primal coefficient (which has no $K$)
    under right-aligned broadcasting.
    """
    base_size = _storage_size(type_cls)
    sb_total = 1
    for s in sub_batch_shape:
        sb_total *= int(s)
    K = sb_total * base_size
    mid = (1,) * len(dyn_shape)
    # eye(K) columns ≅ flat DOF (sub-site × base component); unflatten that axis
    # into (*sub_batch_shape, *BASE_SHAPE) and insert size-1 dynamic batch axes
    # between the leading K and the sub-batch region.
    eye = torch.eye(K, dtype=dtype, device=device)
    eye = eye.reshape(K, *mid, *tuple(sub_batch_shape), *type_cls.BASE_SHAPE)
    data = eye.expand(
        K, *tuple(dyn_shape), *tuple(sub_batch_shape), *type_cls.BASE_SHAPE
    ).contiguous()
    return type_cls(data, sub_batch_ndim=len(sub_batch_shape))


def _tangent_block_to_trailing_k(block: torch.Tensor | TensorWrapper) -> torch.Tensor:
    """Convert a leading-K typed tangent block to the trailing-K raw layout the
    assembly code consumes: ``(*dyn, *sub, n, K)``.

    chain-rule blocks travel as typed wrappers with $K$ as
    the leftmost batch dim ($data.shape == (K, *dyn, *sub, *base)$). The
    matrix-assembly path (``_broadcast_block_to_row_batch`` → ``cat`` → row
    flatten) predates this and operates on trailing-K blocks, so we move $K$
    to the end and flatten the base axes into a single $n$ row dimension. After
    this conversion the existing trailing-K assembly logic is unchanged.
    """
    if not isinstance(block, TensorWrapper):
        return block  # defensive: already a raw trailing-K tensor
    base_ndim = type(block).BASE_NDIM
    moved = block.data.movedim(0, -1)  # (*dyn, *sub, *base, K)
    if base_ndim == 0:
        return moved.unsqueeze(-2)  # (*dyn, *sub, 1, K)
    if base_ndim == 1:
        return moved  # (*dyn, *sub, n, K)
    K = moved.shape[-1]
    return moved.reshape(*moved.shape[: moved.ndim - 1 - base_ndim], -1, K)


class IStructure(Enum):
    """Layout-level structure hint for the solver dispatch.

    Python-side mirror of C++ ``AxisLayout::IStructure`` (see
    ``include/neml2/equation_systems/AxisLayout.h``):

    * ``DENSE`` — no sub-batch structure, or sub-batch shapes don't align
      across variables in the layout, or the residual/unknown
      cross-derivatives are not all sub-batch-diagonal. The system is
      flattened to a dense block and solved with :class:`~solvers.DenseLU`.

    * ``BLOCK`` — every variable in the layout shares the same non-trivial
      sub-batch shape ``(L1, L2, ...)`` AND every (residual, unknown) edge
      is sub-batch-diagonal. The system has block-diagonal structure along
      the sub-batch axes — exactly the case
      :class:`~solvers.SchurComplement` will exploit.

    Today only DENSE is wired through Newton. The BLOCK hint is propagated
    so the future Schur path can dispatch on it.
    """

    DENSE = "dense"
    BLOCK = "block"


def _parse_istructure(token: str) -> IStructure:
    """Resolve a HIT istructure token (case-insensitive) to the enum value."""
    upper = token.upper()
    try:
        return IStructure[upper]
    except KeyError as exc:
        raise ValueError(
            f"Unknown istructure token {token!r}; expected 'BLOCK' or 'DENSE'"
        ) from exc


@dataclass(frozen=True)
class AxisLayout:
    """Ordered variable groups, their tensor types, and optional sub-batch shape.

    Sub-batch handling
    ------------------
    Each variable can carry a ``sub_batch_shape``: the trailing batch axes
    that act as a structured per-site region (see
    :mod:`neml2.types._base`). At construction the layout receives
    a ``sub_batch_shapes`` map; absent or empty entries mean "no sub-batch"
    and default to ``()``.

    Per-group :attr:`group_istructures` declares whether each group has
    block-diagonal sub-batch structure (BLOCK) or not (DENSE). It is inferred
    from ``sub_batch_shapes`` by default; the HIT
    ``istructure = 'BLOCK DENSE'`` option (consumed by
    :class:`ModelNonlinearSystem`) overrides one entry per group. The
    single-value :attr:`istructure` view is the conservative aggregate
    (BLOCK iff every group is BLOCK) and is what older single-group callers
    see — it stays meaningful for the common single-group case while letting
    multi-group :class:`~solvers.SchurComplement` callers inspect each
    group's structure independently.

    :meth:`block_size` returns the per-(dynamic-batch, sub-batch-site)
    storage so a future Schur solver can size its dense per-block reductions
    correctly.
    """

    groups: tuple[tuple[str, ...], ...]
    specs: dict[str, type[TensorWrapper]]
    sub_batch_shapes: dict[str, torch.Size]
    group_istructures: tuple[IStructure, ...]

    def __init__(
        self,
        groups: list[list[str]] | tuple[tuple[str, ...], ...],
        specs: dict[str, type[TensorWrapper]],
        sub_batch_shapes: dict[str, torch.Size] | None = None,
        istructure: IStructure | Sequence[IStructure] | None = None,
    ) -> None:
        normalized = tuple(tuple(group) for group in groups)
        missing = [name for group in normalized for name in group if name not in specs]
        if missing:
            raise KeyError(f"AxisLayout variables missing from specs: {missing}")
        sub = {
            name: torch.Size(sub_batch_shapes.get(name, ()))
            if sub_batch_shapes is not None
            else torch.Size(())
            for group in normalized
            for name in group
        }
        per_group = self._normalize_istructure(istructure, normalized, sub)
        object.__setattr__(self, "groups", normalized)
        object.__setattr__(self, "specs", dict(specs))
        object.__setattr__(self, "sub_batch_shapes", sub)
        object.__setattr__(self, "group_istructures", per_group)

    @staticmethod
    def _normalize_istructure(
        istructure: IStructure | Sequence[IStructure] | None,
        groups: tuple[tuple[str, ...], ...],
        sub_batch_shapes: dict[str, torch.Size],
    ) -> tuple[IStructure, ...]:
        """Resolve the per-group istructure from the optional constructor input."""
        n = len(groups)
        if istructure is None:
            return tuple(
                AxisLayout._infer_group_istructure(group, sub_batch_shapes) for group in groups
            )
        if isinstance(istructure, IStructure):
            return (istructure,) * n
        per_group = tuple(istructure)
        if len(per_group) != n:
            raise ValueError(
                f"AxisLayout istructure must have one entry per group "
                f"(got {len(per_group)}, expected {n})"
            )
        if not all(isinstance(s, IStructure) for s in per_group):
            raise TypeError("AxisLayout istructure entries must be IStructure values")
        return per_group

    @staticmethod
    def _infer_group_istructure(
        group: tuple[str, ...],
        sub_batch_shapes: dict[str, torch.Size],
    ) -> IStructure:
        """BLOCK iff every variable in ``group`` shares the same non-trivial sub-batch shape.

        Conservative default — relies only on shape data. Callers that have
        additional information (e.g. ``Model.list_deriv`` declarations) can
        override by passing ``istructure`` explicitly to the constructor.
        """
        shapes = [sub_batch_shapes[name] for name in group]
        if not shapes:
            return IStructure.DENSE
        first = shapes[0]
        if len(first) == 0:
            return IStructure.DENSE
        return IStructure.BLOCK if all(s == first for s in shapes) else IStructure.DENSE

    @property
    def istructure(self) -> IStructure:
        """Single-value aggregate: BLOCK iff every group is BLOCK, else DENSE.

        Convenience for callers (typically single-group layouts) that don't
        need per-group granularity. Multi-group consumers should read
        :attr:`group_istructures` directly.
        """
        return (
            IStructure.BLOCK
            if all(s == IStructure.BLOCK for s in self.group_istructures)
            else IStructure.DENSE
        )

    def with_sub_batch_shapes(
        self,
        sub_batch_shapes: dict[str, torch.Size],
        *,
        istructure: IStructure | Sequence[IStructure] | None = None,
    ) -> AxisLayout:
        """Return a new layout with updated sub-batch shapes (frozen replacement)."""
        return AxisLayout(self.groups, self.specs, sub_batch_shapes, istructure)

    def sub_layout(self, index: int) -> AxisLayout:
        """Single-group sub-layout containing only ``self.groups[index]``.

        Backs the :meth:`AssembledVector.group` / :meth:`AssembledMatrix.group`
        block-view helpers used by :class:`~solvers.SchurComplement`. The new
        layout's istructure is the parent's per-group entry at ``index`` —
        preserves BLOCK/DENSE for that specific group.
        """
        group = self.groups[index]
        specs = {name: self.specs[name] for name in group}
        sub_batch = {name: self.sub_batch_shapes[name] for name in group}
        return AxisLayout([list(group)], specs, sub_batch, self.group_istructures[index])

    @property
    def ngroup(self) -> int:
        return len(self.groups)

    @property
    def nvar(self) -> int:
        return sum(len(group) for group in self.groups)

    def vars(self) -> tuple[str, ...]:
        return tuple(name for group in self.groups for name in group)

    def group_size(self, index: int) -> int:
        return sum(self.var_size(name) for name in self.groups[index])

    def storage_size(self) -> int:
        return sum(self.group_size(i) for i in range(self.ngroup))

    def var_size(self, name: str) -> int:
        return _storage_size(self.specs[name])

    def sub_batch_shape(self, name: str) -> torch.Size:
        """Per-variable sub-batch shape (empty when the var is sub-batch-trivial)."""
        return self.sub_batch_shapes.get(name, torch.Size(()))

    def block_size(self) -> int:
        """Per-(dynamic-batch, sub-batch-site) storage size.

        For ``IStructure.DENSE`` this equals :meth:`storage_size`. For
        ``IStructure.BLOCK`` this is the storage *inside* one sub-batch
        site, which the Schur path will treat as the dense reduction
        unit. Today this is used only as a sanity-check primitive — no
        solver consumes it yet.
        """
        return self.storage_size()

    def type_of(self, name: str) -> type[TensorWrapper]:
        return self.specs[name]


@dataclass
class AssembledVector:
    """Dense vector blocks assembled by variable group."""

    layout: AxisLayout
    tensors: list[torch.Tensor]

    @classmethod
    def from_dict(cls, layout: AxisLayout, values: dict[str, torch.Tensor]) -> AssembledVector:
        """Pack ``values`` into per-group flat tensors.

        Each variable's ``(*dyn, *sub_batch, *base)`` storage gets flattened
        to ``(*dyn, prod(sub_batch) * base_size)``, then concatenated across
        variables within a group at the trailing axis. This keeps the
        assembled vector shape uniform per group regardless of whether the
        group's variables carry sub-batch dimensions.

        ``*dyn`` is inferred per-value as ``value.ndim − sub_batch_ndim −
        BASE_NDIM`` so the helper works on values regardless of how many
        dynamic-batch axes they carry.
        """
        tensors: list[torch.Tensor] = []
        for group in layout.groups:
            parts: list[torch.Tensor] = []
            for name in group:
                type_cls = layout.type_of(name)
                sb_shape = layout.sub_batch_shape(name)
                value = values[name]
                # dyn_ndim = total batch ndim - sub_batch ndim.
                batch_ndim = value.ndim - type_cls.BASE_NDIM
                dyn_ndim = batch_ndim - len(sb_shape)
                parts.append(_flatten_sub_batch_and_base(value, type_cls, sb_shape, dyn_ndim))
            if len(parts) > 1:
                # Parts in the same group may have different dyn rank when
                # one variable was broadcast up by a per-step-batched primal
                # and another wasn't (e.g. ``y_residual`` shape ``(1, 6)``
                # alongside a base-shape ``target_cauchy_stress_residual``
                # shape ``(6,)``). Broadcast all parts to a common leading
                # shape before catting along the trailing flat dim.
                lead_shape = torch.broadcast_shapes(*[p.shape[:-1] for p in parts])
                parts = [p.broadcast_to((*lead_shape, p.shape[-1])) for p in parts]
                tensors.append(torch.cat(parts, dim=-1))
            else:
                tensors.append(parts[0])
        return cls(layout, tensors)

    def disassemble(self) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`from_dict` — recovers per-variable shapes.

        Splits each group's flat tensor at the per-variable boundaries
        (``prod(sub_batch_shape) * base_size`` each) and reshapes back to
        ``(*dyn, *sub_batch, *base)``.
        """
        values: dict[str, torch.Tensor] = {}
        for group, tensor in zip(self.layout.groups, self.tensors, strict=True):
            offset = 0
            for name in group:
                type_cls = self.layout.type_of(name)
                sb_shape = self.layout.sub_batch_shape(name)
                sb_total = 1
                for s in sb_shape:
                    sb_total *= int(s)
                size = sb_total * self.layout.var_size(name)
                # dyn_ndim is anything before the trailing flat axis in `tensor`.
                dyn_ndim = tensor.ndim - 1
                slab = tensor[..., offset : offset + size]
                values[name] = _unflatten_sub_batch_and_base(slab, type_cls, sb_shape, dyn_ndim)
                offset += size
        return values

    def __neg__(self) -> AssembledVector:
        return AssembledVector(self.layout, [-tensor for tensor in self.tensors])

    def __add__(self, other: AssembledVector) -> AssembledVector:
        if self.layout != other.layout:
            raise ValueError("Cannot add AssembledVector objects with different layouts")
        return AssembledVector(
            self.layout, [a + b for a, b in zip(self.tensors, other.tensors, strict=True)]
        )

    def __sub__(self, other: AssembledVector) -> AssembledVector:
        return self + (-other)

    def group(self, index: int) -> AssembledVector:
        """Single-group sub-vector view of ``self.tensors[index]``.

        Returns a fresh :class:`AssembledVector` whose layout contains only the
        ``index``-th group of ``self.layout``. No tensor copy — the underlying
        block is shared. Used by :class:`~solvers.SchurComplement` to extract
        $b_p$ / $b_s$ for the two-group Schur factorisation.
        """
        return AssembledVector(self.layout.sub_layout(index), [self.tensors[index]])


@dataclass
class AssembledMatrix:
    """Dense matrix blocks assembled by row and column variable groups.

    ``_block_dict`` is an optional name-keyed cache of the per-(row_var,
    col_var) tangent blocks that fed assembly — populated by
    :meth:`ModelNonlinearSystem._assemble_matrix` and consumed by
    :meth:`disassemble`. External callers do not need to set it; matrices
    constructed by arithmetic (``__neg__``, ``__sub__``, ``__matmul__``)
    propagate ``None`` because the per-block identities don't survive
    block-wise combination.
    """

    row_layout: AxisLayout
    col_layout: AxisLayout
    tensors: list[list[torch.Tensor]]
    _block_dict: dict[str, dict[str, torch.Tensor]] | None = field(
        default=None, repr=False, compare=False
    )

    def __neg__(self) -> AssembledMatrix:
        return AssembledMatrix(
            self.row_layout,
            self.col_layout,
            [[-block for block in row] for row in self.tensors],
        )

    def __sub__(self, other: AssembledMatrix) -> AssembledMatrix:
        if self.row_layout != other.row_layout or self.col_layout != other.col_layout:
            raise ValueError("Cannot subtract AssembledMatrix objects with different layouts")
        return AssembledMatrix(
            self.row_layout,
            self.col_layout,
            [
                [a - b for a, b in zip(row_a, row_b, strict=True)]
                for row_a, row_b in zip(self.tensors, other.tensors, strict=True)
            ],
        )

    def group(self, i: int, j: int) -> AssembledMatrix:
        """Single-block sub-matrix view of ``self.tensors[i][j]``.

        Returns a 1×1 block :class:`AssembledMatrix` whose row layout contains
        only row group ``i`` and whose column layout contains only column
        group ``j``. No tensor copy. Used by
        :class:`~solvers.SchurComplement` to extract the four ``A_pp / A_ps /
        A_sp / A_ss`` blocks for the Schur factorisation.
        """
        return AssembledMatrix(
            self.row_layout.sub_layout(i),
            self.col_layout.sub_layout(j),
            [[self.tensors[i][j]]],
        )

    def disassemble(self) -> dict[str, dict[str, torch.Tensor]]:
        """Per-(row_var, col_var) chain-rule blocks as a name-keyed nested dict.

        Each block has shape ``(*row_dyn, *row_sub, row_n_flat, col_K)`` where
        $col_K = prod(col_sub) * col_n_flat$. The exact shape that
        :meth:`select_blocks` consumes, so
        ``AssembledMatrix.select_blocks(rlayout, clayout, M.disassemble())``
        rebuilds an equivalent assembled tensor.

        Only matrices produced by :meth:`LinearSystem.assemble` carry the
        per-block cache; matrices constructed by arithmetic (``__neg__``,
        ``__sub__``, ``__matmul__``) lose it because block identities do not
        survive block-wise combination. Calling :meth:`disassemble` on such a
        matrix raises :exc:`RuntimeError`.
        """
        if self._block_dict is None:
            raise RuntimeError(
                "AssembledMatrix.disassemble: this matrix was built without a "
                "per-block cache (typically because it came from arithmetic "
                "rather than directly from LinearSystem.assemble). Recompute "
                "by re-running assemble() on the source system."
            )
        return {row: dict(cols) for row, cols in self._block_dict.items()}

    @classmethod
    def select_blocks(
        cls,
        row_layout: AxisLayout,
        col_layout: AxisLayout,
        blocks: dict[str, dict[str, torch.Tensor]],
    ) -> AssembledMatrix:
        """Assemble a fresh :class:`AssembledMatrix` from a name-keyed lookup.

        Replaces the C++-side ``SparseMatrix(rlayout, clayout, [[...]]).assemble()``
        round-trip used by the pyzag interface. Each ``blocks[row_var][col_var]``
        tensor must have the same shape contract :meth:`disassemble` produces —
        ``(*row_dyn, *row_sub, row_n_flat, col_K)``. Missing pairs are filled
        with zeros sized from ``col_layout.var_size(col_name) *
        prod(col_layout.sub_batch_shape(col_name))`` at the row's batch shape.

        Per row group there must be at least one present block somewhere
        (under any row name in the group) so the per-row dynamic-batch shape
        and dtype/device can be inferred — pyzag's selection patterns always
        satisfy this. Raises :exc:`ValueError` otherwise.
        """
        col_K: dict[str, int] = {}
        for group in col_layout.groups:
            for name in group:
                sb = col_layout.sub_batch_shape(name)
                sb_total = 1
                for s in sb:
                    sb_total *= int(s)
                col_K[name] = sb_total * col_layout.var_size(name)

        rows: list[list[torch.Tensor]] = []
        for row_group in row_layout.groups:
            group_ref: torch.Tensor | None = None
            for row_name in row_group:
                for present in blocks.get(row_name, {}).values():
                    group_ref = present
                    break
                if group_ref is not None:
                    break
            if group_ref is None:
                raise ValueError(
                    "select_blocks: no present blocks for any row in group "
                    f"{list(row_group)}; cannot infer batch shape or dtype/device."
                )

            block_row: list[torch.Tensor] = []
            for col_group in col_layout.groups:
                row_parts: list[torch.Tensor] = []
                for row_name in row_group:
                    row_sb = row_layout.sub_batch_shape(row_name)
                    row_n = row_layout.var_size(row_name)
                    # Per-row reference falls back to the group ref when this
                    # specific row_name has no entries — gives us dtype/device
                    # plus a shape to derive row_dyn from.
                    row_ref: torch.Tensor | None = None
                    for present in blocks.get(row_name, {}).values():
                        row_ref = present
                        break
                    ref = row_ref if row_ref is not None else group_ref
                    # ref shape: (*dyn, *sub, n_row, K). row_dyn slice is
                    # ref.ndim - 2 - len(row_sb) leading dims.
                    row_dyn_ndim = ref.ndim - 2 - len(row_sb)
                    if row_dyn_ndim < 0:
                        raise ValueError(
                            f"select_blocks: reference block for row {row_name!r} "
                            f"has ndim={ref.ndim} but row_sub_batch shape "
                            f"{tuple(row_sb)} expects at least {2 + len(row_sb)} dims."
                        )
                    row_batch_dyn = tuple(ref.shape[:row_dyn_ndim])
                    row_batch_full = row_batch_dyn + tuple(row_sb)
                    col_parts: list[torch.Tensor] = []
                    for col_name in col_group:
                        block = blocks.get(row_name, {}).get(col_name)
                        if block is None:
                            block = ref.new_zeros(*row_batch_full, row_n, col_K[col_name])
                        col_parts.append(block)
                    cat_cols = torch.cat(col_parts, dim=-1)
                    K_total = cat_cols.shape[-1]
                    cat_cols = cat_cols.reshape(*row_batch_dyn, -1, K_total)
                    row_parts.append(cat_cols)
                if len(row_parts) > 1:
                    lead = torch.broadcast_shapes(*[p.shape[:-2] for p in row_parts])
                    row_parts = [p.broadcast_to((*lead, *p.shape[-2:])) for p in row_parts]
                block_row.append(torch.cat(row_parts, dim=-2))
            rows.append(block_row)
        return cls(row_layout, col_layout, rows)

    @overload
    def __matmul__(self, other: AssembledVector) -> AssembledVector: ...
    @overload
    def __matmul__(self, other: AssembledMatrix) -> AssembledMatrix: ...
    def __matmul__(self, other):
        """Block-wise ``A @ B`` (matrix RHS) or ``A @ b`` (vector RHS).

        Mirrors the C++ ``operator*`` overloads used in
        :class:`SchurComplement`. The inner block dimension is contracted via
        :func:`torch.matmul`, which broadcasts sub-batch axes naturally; the
        result's outer block layout is ``(self.row_layout, other.col_layout)``
        for the matrix RHS or ``self.row_layout`` for the vector RHS.

        The middle layout — ``self.col_layout`` vs. ``other.row_layout`` /
        ``other.layout`` — must match (each block ``self.tensors[i][k]``
        contracts with ``other.tensors[k][j]`` /  ``other.tensors[k]``).
        """
        if isinstance(other, AssembledVector):
            if self.col_layout != other.layout:
                raise ValueError(
                    "AssembledMatrix @ AssembledVector requires matching inner layouts"
                )
            n_rows = self.row_layout.ngroup
            n_inner = self.col_layout.ngroup
            out_tensors: list[torch.Tensor] = []
            for i in range(n_rows):
                acc: torch.Tensor | None = None
                for k in range(n_inner):
                    # (*B, m, n) @ (*B, n) -> (*B, m): unsqueeze RHS to a column.
                    contrib = torch.matmul(
                        self.tensors[i][k], other.tensors[k].unsqueeze(-1)
                    ).squeeze(-1)
                    acc = contrib if acc is None else acc + contrib
                assert acc is not None
                out_tensors.append(acc)
            return AssembledVector(self.row_layout, out_tensors)

        if self.col_layout != other.row_layout:
            raise ValueError("AssembledMatrix @ AssembledMatrix requires matching inner layouts")
        n_rows = self.row_layout.ngroup
        n_inner = self.col_layout.ngroup
        n_cols = other.col_layout.ngroup
        out_blocks: list[list[torch.Tensor]] = []
        for i in range(n_rows):
            row_blocks: list[torch.Tensor] = []
            for j in range(n_cols):
                acc: torch.Tensor | None = None
                for k in range(n_inner):
                    contrib = torch.matmul(self.tensors[i][k], other.tensors[k][j])
                    acc = contrib if acc is None else acc + contrib
                assert acc is not None
                row_blocks.append(acc)
            out_blocks.append(row_blocks)
        return AssembledMatrix(self.row_layout, other.col_layout, out_blocks)


def _broadcast_block_to_row_batch(
    block: torch.Tensor, target_batch_shape: tuple[int, ...]
) -> torch.Tensor:
    """Broadcast a chain-rule block ``(*B_block, n_row, n_col)`` to
    ``(*target_batch_shape, n_row, n_col)``.

    Needed for ``BLOCK + DENSE`` equation systems (per-crystal vs global
    sub-batch): chain rules from global upstream inputs through per-crystal
    residuals produce blocks missing the per-crystal sub-batch axis.
    Conversely, with placeholder-padded identity seeds (see
    :meth:`ModelNonlinearSystem._identity_seed`), chain-rule blocks can
    carry leading size-1 dyn axes that the row residual lacks — those
    placeholders are squeezed here. In both cases the block's batch axes
    are aligned with the target by inserting / removing size-1 leading
    dimensions just before the trailing ``(n_row, n_col)`` pair.
    """
    n_row, n_col = block.shape[-2:]
    cur_batch = tuple(block.shape[:-2])
    target = tuple(target_batch_shape)
    if cur_batch == target:
        return block
    # Trim leading size-1 batch axes the target doesn't carry.
    while len(cur_batch) > len(target):
        if cur_batch[0] != 1:
            raise ValueError(
                f"chain-rule block has a non-singleton extra batch axis "
                f"({cur_batch}) that the target residual ({target}) cannot accept."
            )
        block = block.squeeze(0)
        cur_batch = cur_batch[1:]
    if cur_batch == target:
        return block.contiguous()
    # Pad missing leading batch axes with size-1, then broadcast.
    extra = len(target) - len(cur_batch)
    new_block = block
    for _ in range(extra):
        new_block = new_block.unsqueeze(-3)
    return new_block.expand(*target, n_row, n_col).contiguous()


def norm(v: AssembledVector) -> torch.Tensor:
    """Batched Euclidean norm over all assembled vector groups."""

    return torch.sqrt(norm_sq(v))


def norm_sq(v: AssembledVector) -> torch.Tensor:
    """Batched squared Euclidean norm over all assembled vector groups.

    Mirrors C++ ``neml2::norm_sq``; used by ``NewtonWithLineSearch`` to avoid
    the sqrt in the per-element line-search criterion.
    """
    total: torch.Tensor | None = None
    for tensor in v.tensors:
        contribution = (tensor * tensor).sum(dim=-1)
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("Cannot compute norm of an empty AssembledVector")
    return total


class LinearSystem:
    """Base class for systems with assembled operators."""

    #: HIT section for ``neml2-syntax`` classification — inherited by every
    #: registered subclass (``ModelNonlinearSystem`` lives under
    #: ``[EquationSystems]`` in the input file).
    SECTION = "EquationSystems"

    def __init__(self) -> None:
        self._ulayout = self.setup_ulayout()
        self._glayout = self.setup_glayout()
        self._blayout = self.setup_blayout()

    @property
    def ulayout(self) -> AxisLayout:
        return self._ulayout

    @property
    def glayout(self) -> AxisLayout:
        return self._glayout

    @property
    def blayout(self) -> AxisLayout:
        return self._blayout

    def setup_ulayout(self) -> AxisLayout:
        raise NotImplementedError

    def setup_glayout(self) -> AxisLayout:
        raise NotImplementedError

    def setup_blayout(self) -> AxisLayout:
        raise NotImplementedError

    def set_u(self, u: AssembledVector) -> None:
        raise NotImplementedError

    def set_g(self, g: AssembledVector) -> None:
        raise NotImplementedError

    def u(self) -> AssembledVector:
        raise NotImplementedError

    def g(self) -> AssembledVector:
        raise NotImplementedError

    def assemble(
        self,
        need_A: bool,
        need_B: bool,
        need_b: bool,
    ) -> tuple[AssembledMatrix | None, AssembledMatrix | None, AssembledVector | None]:
        raise NotImplementedError

    def A(self) -> AssembledMatrix:
        A, _, _ = self.assemble(True, False, False)
        assert A is not None
        return A

    def b(self) -> AssembledVector:
        _, _, b = self.assemble(False, False, True)
        assert b is not None
        return b

    def A_and_b(self) -> tuple[AssembledMatrix, AssembledVector]:
        A, _, b = self.assemble(True, False, True)
        assert A is not None and b is not None
        return A, b

    def A_and_B(self) -> tuple[AssembledMatrix, AssembledMatrix]:
        A, B, _ = self.assemble(True, True, False)
        assert A is not None and B is not None
        return A, B

    def A_and_B_and_b(self) -> tuple[AssembledMatrix, AssembledMatrix, AssembledVector]:
        A, B, b = self.assemble(True, True, True)
        assert A is not None and B is not None and b is not None
        return A, B, b


class NonlinearSystem(LinearSystem):
    """Nonlinear system with C++-matching Newton sign convention."""


@register_native("NonlinearSystem")
class ModelNonlinearSystem(NonlinearSystem):
    """A nonlinear system defined by a Model."""

    hit = HitSchema(
        dependency("model", "get_model", "The Model defining this nonlinear system."),
        option(
            "unknowns",
            list,
            "Ordering and grouping of unknowns. Each inner list defines one variable group.",
        ),
        option(
            "residuals",
            list,
            "Ordering and grouping of residual variables. Each inner list defines one variable "
            "group.",
            default=[],
        ),
        option(
            "istructure",
            list,
            "Optional IStructure for each variable group. If not provided, defaults to DENSE. If "
            "only one IStructure is provided, it will be applied to all groups.",
            default=[],
        ),
    )

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> ModelNonlinearSystem:
        model = factory.get_model(node.param_str("model"))
        # HIT `unknowns = 'a b; c'` ⇒ two groups [['a','b'], ['c']]; a single
        # whitespace-separated list with no `;` collapses to one group. Matches
        # the C++ side (which also uses `;` as the group separator) and lets
        # SchurComplement-equipped inputs cross-parse identically.
        unknowns = [list(group) for group in node.param_list_list_str("unknowns")]
        residuals: list[list[str]] | None = None
        if node.find("residuals") is not None:
            residuals = [list(group) for group in node.param_list_list_str("residuals")]
        # Optional per-group `istructure = 'BLOCK DENSE'` override (one entry
        # per unknown/residual group). Skipped when absent — AxisLayout falls
        # back to shape-based inference per group.
        istructure: list[IStructure] | None = None
        if node.find("istructure") is not None:
            istructure = [_parse_istructure(s) for s in node.param_list_str("istructure")]
        return cls(model, unknowns=unknowns, residuals=residuals, istructure=istructure)

    def __init__(
        self,
        model: Model,
        unknowns: list[list[str]],
        residuals: list[list[str]] | None = None,
        istructure: Sequence[IStructure] | None = None,
    ) -> None:
        self.model = model
        self.unknown_groups = [list(group) for group in unknowns]
        self.residual_groups = (
            [list(group) for group in residuals]
            if residuals is not None
            else self._infer_residual_groups(self.unknown_groups)
        )
        self.unknown_names = [name for group in self.unknown_groups for name in group]
        self.residual_names = [name for group in self.residual_groups for name in group]
        self.given_names = [name for name in model.input_spec if name not in self.unknown_names]
        if istructure is not None and len(istructure) != len(self.unknown_groups):
            raise ValueError(
                f"istructure must have one entry per unknown group "
                f"(got {len(istructure)}, expected {len(self.unknown_groups)})"
            )
        self._istructure = tuple(istructure) if istructure is not None else None
        self._state: dict[str, torch.Tensor] = {}
        # Populated deterministically at ``initialize`` time from the
        # caller-supplied per-variable ``sub_batch_ndim`` dict. Until then
        # both dicts are empty — every consumer site reads them per-name and
        # treats absence as "no sub-batch".
        self._dynamic_batch_ndim: dict[str, int] = {}
        self._sub_batch_shapes: dict[str, torch.Size] = {}
        super().__init__()

    def _infer_residual_groups(self, unknowns: list[list[str]]) -> list[list[str]]:
        # Default residual-name convention matches C++ `residual_name()`:
        # `<variable>_residual`. The HIT `residuals = '...'` option overrides
        # this for cases where the convention doesn't apply (e.g. the J2
        # complementarity residual is named `complementarity`, not
        # `flow_rate_residual`).
        residual_groups: list[list[str]] = []
        for group in unknowns:
            residual_group: list[str] = []
            for name in group:
                residual = f"{name}_residual"
                if residual not in self.model.output_spec:
                    raise KeyError(
                        f"Could not infer residual for unknown {name!r}; "
                        f"expected output {residual!r}. Provide explicit "
                        "`residuals = ...` in the [EquationSystems] block."
                    )
                residual_group.append(residual)
            residual_groups.append(residual_group)
        return residual_groups

    def setup_ulayout(self) -> AxisLayout:
        return AxisLayout(
            self.unknown_groups,
            self.model.input_spec,
            sub_batch_shapes={
                k: v for k, v in self._sub_batch_shapes.items() if k in self.unknown_names
            },
            istructure=self._istructure,
        )

    def setup_glayout(self) -> AxisLayout:
        return AxisLayout(
            [self.given_names],
            self.model.input_spec,
            sub_batch_shapes={
                k: v for k, v in self._sub_batch_shapes.items() if k in self.given_names
            },
        )

    def setup_blayout(self) -> AxisLayout:
        # Residuals inherit per-variable sub-batch from the matching unknown
        # (residual_groups[i][j] corresponds to unknown_groups[i][j]).
        residual_sb: dict[str, torch.Size] = {}
        for ugroup, rgroup in zip(self.unknown_groups, self.residual_groups, strict=True):
            for uname, rname in zip(ugroup, rgroup, strict=True):
                if uname in self._sub_batch_shapes:
                    residual_sb[rname] = self._sub_batch_shapes[uname]
        return AxisLayout(
            self.residual_groups,
            self.model.output_spec,
            sub_batch_shapes=residual_sb,
            istructure=self._istructure,
        )

    def set_u(self, u: AssembledVector) -> None:
        self._state.update(u.disassemble())

    def set_g(self, g: AssembledVector) -> None:
        self._state.update(g.disassemble())

    def u(self) -> AssembledVector:
        return AssembledVector.from_dict(self.ulayout, self._state)

    def g(self) -> AssembledVector:
        return AssembledVector.from_dict(self.glayout, self._state)

    def initialize(
        self,
        *,
        u: dict[str, torch.Tensor],
        g: dict[str, torch.Tensor],
        sub_batch_ndim: dict[str, int] | None = None,
        dyn_shape: tuple[int, ...] = (),
    ) -> None:
        """Set the state and derive per-variable layout from caller-declared sub_batch_ndim.

        ``sub_batch_ndim[name]`` is the deterministic source of truth: each
        typed wrapper upstream carries its own ``sub_batch_ndim`` (set by the
        Driver / IC declaration / composed-model propagation), and the caller
        (typically :class:`~models.common.ImplicitUpdate`) forwards that
        per-variable dict here. Variables missing from the dict (or when the
        dict itself is ``None`` — direct-test callers) are treated as
        ``sub_batch_ndim == 0`` — explicit "nothing declared sub-batch," which
        matches the observable behavior for every existing single-group
        scenario (their leading batch axes are already pure dyn).
        """
        self._state = {**g, **u}
        # Caller-declared system-wide dynamic-batch shape. Drives identity-seed
        # padding so the seed's dyn axes align with primal values in chain-rule
        # ops — replaces the earlier ``max(state value shapes)`` heuristic
        # which broke once Newton mutated state shapes during iteration.
        self._dyn_shape: tuple[int, ...] = tuple(dyn_shape)

        self._sub_batch_shapes = {}
        self._dynamic_batch_ndim = {}
        for name, value in self._state.items():
            type_cls = self.model.input_spec[name]
            batch = _batch_shape(value, type_cls)
            sbn = sub_batch_ndim.get(name, 0) if sub_batch_ndim is not None else 0
            if sbn > 0:
                if sbn > len(batch):
                    raise ValueError(
                        f"sub_batch_ndim[{name!r}]={sbn} exceeds batch_ndim={len(batch)} "
                        f"of value with shape {tuple(value.shape)}"
                    )
                self._sub_batch_shapes[name] = torch.Size(batch[-sbn:])
            self._dynamic_batch_ndim[name] = len(batch) - sbn

        # Mirror per-unknown sub-batch + dyn-ndim onto the matching residual
        # (residual_groups[i][j] corresponds to unknown_groups[i][j]). Both
        # ``setup_blayout`` and ``_assemble_matrix`` key by residual name, so
        # the dicts must carry residual entries too.
        for ugroup, rgroup in zip(self.unknown_groups, self.residual_groups, strict=True):
            for uname, rname in zip(ugroup, rgroup, strict=True):
                if uname in self._sub_batch_shapes:
                    self._sub_batch_shapes[rname] = self._sub_batch_shapes[uname]
                self._dynamic_batch_ndim[rname] = self._dynamic_batch_ndim[uname]

        # Rebuild layouts so they carry the declared sub-batch shapes (used
        # by the AssembledVector flatten/unflatten paths and the per-block
        # row flatten in _assemble_matrix).
        self._ulayout = self.setup_ulayout()
        self._glayout = self.setup_glayout()
        self._blayout = self.setup_blayout()

    def _call_model(self, v: ChainRuleDict | None = None) -> tuple[Any, ...]:
        missing = [name for name in self.model.input_spec if name not in self._state]
        if missing:
            raise KeyError(f"ModelNonlinearSystem state is missing inputs: {missing}")
        # Wrap each raw input as a typed wrapper carrying the system-inferred
        # sub_batch_ndim. Without this the downstream
        # ``ComposedModel`` would see ``sub_batch_ndim=0`` even for variables
        # the system knows to be per-sub-batch-site (per D-052), forcing
        # every leaf to fall back on ``.data`` + manual middle-padding.
        args = tuple(
            self.model.input_spec[name](
                self._state[name],
                sub_batch_ndim=len(self._sub_batch_shapes.get(name, ())),
            )
            for name in self.model.input_spec
        )
        result = self.model(*args, v=v) if v is not None else self.model(*args)
        return result if isinstance(result, tuple) else (result,)

    def _identity_seed(self, names: list[str]) -> ChainRuleDict:
        """Per-crystal-expanded identity seed.

        For each unknown's per-(dynamic-batch, sub-batch-site, base) DOF a
        distinct column is produced — preserves per-sub-batch-site
        Jacobian columns through reductions like ``SR2IntermediateMean``
        (which would otherwise collapse them).

        Seeds for variables whose state value has fewer dynamic-batch axes
        than the system's max are LEFT-padded with size-1 placeholders so
        the leading-K tangent contract ``(K, *dyn, *sub, *base)`` lines up
        position-wise across all variables. Without these placeholders, a
        tangent for a base-shape-only unknown (e.g. ``flow_rate`` at the
        first elastic step) would have shape ``(K, base)`` and collide
        positionally with a force primal's ``(dyn, base)`` during the
        typed-wrapper algebra — torch's right-aligned broadcast would
        then misalign K with dyn. The size-1 placeholders make the
        broadcast structurally correct so the model's chain-rule
        propagation works identically whether the unknown starts
        base-shape or per-step-batched.
        """
        # Caller-declared system-wide dynamic-batch shape is the source of
        # truth (set in :meth:`initialize`). Each seed is sized to align
        # against this dyn shape: per-var ``own_dyn`` from the current state
        # value's actual leading axes, then left-padded with size-1
        # placeholders up to ``len(self._dyn_shape)`` so the leading-K
        # tangent contract ``(K, *dyn, *sub, *base)`` lines up
        # position-wise across all variables. No state-shape-derived
        # heuristics — the caller (typically ImplicitUpdate) tells the
        # system what dyn shape the scenario uses, and the system trusts
        # that declaration through every Newton iteration regardless of
        # how the unknown's shape evolves.
        target_dyn_ndim = len(self._dyn_shape)
        seed: ChainRuleDict = {}
        for name in names:
            type_cls = self.model.input_spec[name]
            value = self._state[name]
            sub_batch_shape = tuple(self._sub_batch_shapes.get(name, ()))
            # Per-variable live dyn_ndim from the current state value's
            # actual shape (handles base-shape-only initial guesses that
            # later expand to per-step batched after a Newton update).
            live_dyn_ndim = max(value.ndim - type_cls.BASE_NDIM - len(sub_batch_shape), 0)
            own_dyn = tuple(_batch_shape(value, type_cls)[:live_dyn_ndim])
            # Left-pad with size-1 placeholders to reach the target ndim.
            pad = target_dyn_ndim - len(own_dyn)
            dyn_shape = ((1,) * pad if pad > 0 else ()) + own_dyn
            seed[name] = {
                name: _expanded_identity_seed(
                    type_cls,
                    sub_batch_shape,
                    dyn_shape,
                    dtype=value.dtype,
                    device=value.device,
                )
            }
        return seed

    def _zero_block(
        self,
        residual_name: str,
        input_name: str,
        *,
        col_K: int,
        row_batch: tuple[int, ...],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        row_size = _storage_size(self.model.output_spec[residual_name])
        return ref.new_zeros(*row_batch, row_size, col_K)

    def _assemble_matrix(
        self,
        row_layout: AxisLayout,
        col_layout: AxisLayout,
        v_out: ChainRuleDict,
        like_by_row: dict[str, torch.Tensor],
    ) -> AssembledMatrix:
        # Build per-column K (expanded) once per col_name so zero blocks
        # match the chain-rule block's K. ``col_K[name] = prod(sub_batch_shape) * base_size``.
        col_K: dict[str, int] = {}
        for group in col_layout.groups:
            for name in group:
                sb = col_layout.sub_batch_shape(name)
                sb_total = 1
                for s in sb:
                    sb_total *= int(s)
                col_K[name] = sb_total * col_layout.var_size(name)

        # Per-(row_var, col_var) block cache. Each entry has shape
        # ``(*row_dyn, *row_sub, row_n_flat, col_K)`` — matches what
        # ``AssembledMatrix.select_blocks`` consumes, so a disassemble →
        # select_blocks round-trip recovers the same assembled tensors.
        block_dict: dict[str, dict[str, torch.Tensor]] = {}

        rows: list[list[torch.Tensor]] = []
        for row_group in row_layout.groups:
            block_row: list[torch.Tensor] = []
            for col_group in col_layout.groups:
                row_parts: list[torch.Tensor] = []
                for row_name in row_group:
                    row_sb = row_layout.sub_batch_shape(row_name)
                    # Infer dyn from the ACTUAL residual value's shape, not
                    # from ``self._dynamic_batch_ndim`` (which is per-unknown,
                    # state-value-derived). When an unknown starts at
                    # base-shape but the residual broadcasts up to the
                    # per-step batch, the actual residual ndim is the right
                    # reference for slicing the leading-batch portion.
                    # ``like_by_row[row_name]`` has shape
                    # ``(*dyn, *sub, n_row_flat)`` after ``_flatten_base``.
                    row_total_ndim = like_by_row[row_name].ndim - 1
                    row_dyn_ndim = row_total_ndim - len(row_sb)
                    row_batch_dyn = like_by_row[row_name].shape[:row_dyn_ndim]
                    row_batch_full = tuple(row_batch_dyn) + tuple(row_sb)
                    col_parts: list[torch.Tensor] = []
                    for col_name in col_group:
                        t_block = v_out.get(row_name, {}).get(col_name)
                        if t_block is None:
                            block = self._zero_block(
                                row_name,
                                col_name,
                                col_K=col_K[col_name],
                                row_batch=row_batch_full,
                                ref=like_by_row[row_name],
                            )
                        else:
                            # chain-rule blocks are leading-K
                            # typed wrappers; convert to the trailing-K raw
                            # layout the AssembledMatrix path consumes.
                            block = _broadcast_block_to_row_batch(
                                _tangent_block_to_trailing_k(t_block),
                                row_batch_full,
                            )
                        col_parts.append(block)
                        block_dict.setdefault(row_name, {})[col_name] = block
                    # Cat per-row-name's col blocks along the K (col) dim.
                    cat_cols = torch.cat(col_parts, dim=-1)
                    # Flatten the per-row sub_batch axes into the row dim.
                    # cat_cols shape: (*dyn, *row_sb, n_row, K_total). Folding
                    # (*row_sb, n_row) → n_row * row_sb_total via reshape with
                    # an explicit -1 so we don't need to spell either out.
                    K_total = cat_cols.shape[-1]
                    cat_cols = cat_cols.reshape(*row_batch_dyn, -1, K_total)
                    # Now (n_row * row_sb_total) at dim -2; n_row depends on
                    # this row_name only. Append at the row dim.
                    row_parts.append(cat_cols)
                # Cat across row_names in the row_group at the row dim. Row
                # vars in the same group may have different leading-dyn rank
                # (e.g. one residual was broadcast up by a per-step force,
                # another wasn't) — broadcast to a common leading shape so
                # the trailing-row cat aligns.
                if len(row_parts) > 1:
                    lead = torch.broadcast_shapes(*[p.shape[:-2] for p in row_parts])
                    row_parts = [p.broadcast_to((*lead, *p.shape[-2:])) for p in row_parts]
                block_row.append(torch.cat(row_parts, dim=-2))
            rows.append(block_row)
        return AssembledMatrix(row_layout, col_layout, rows, _block_dict=block_dict)

    def assemble(
        self,
        need_A: bool,
        need_B: bool,
        need_b: bool,
    ) -> tuple[AssembledMatrix | None, AssembledMatrix | None, AssembledVector | None]:
        seed_names: list[str] = []
        if need_A:
            seed_names.extend(self.unknown_names)
        if need_B:
            seed_names.extend(self.given_names)
        seed = self._identity_seed(seed_names) if seed_names else None
        result = self._call_model(v=seed)

        if seed is None:
            output_values = result
            v_out: ChainRuleDict = {}
        else:
            output_values = result[:-1]
            v_out = result[-1]

        output_state = {
            name: value.data if isinstance(value, TensorWrapper) else value
            for name, value in zip(self.model.output_spec, output_values, strict=True)
        }
        residual_values = {name: output_state[name] for name in self.residual_names}
        like_by_row = {
            name: _flatten_base(residual_values[name], self.model.output_spec[name])
            for name in self.residual_names
        }

        A = (
            self._assemble_matrix(self.blayout, self.ulayout, v_out, like_by_row)
            if need_A
            else None
        )
        B = (
            self._assemble_matrix(self.blayout, self.glayout, v_out, like_by_row)
            if need_B
            else None
        )
        b = -AssembledVector.from_dict(self.blayout, residual_values) if need_b else None
        return A, B, b


class _DenseSystemModule(nn.Module):
    """Tensor-only dense export surface for a frozen `ModelNonlinearSystem` layout."""

    def __init__(self, system: ModelNonlinearSystem) -> None:
        super().__init__()
        self.model = system.model
        self.ulayout = system.ulayout
        self.glayout = system.glayout
        self.blayout = system.blayout
        self.unknown_names = tuple(system.unknown_names)
        self.given_names = tuple(system.given_names)
        self.residual_names = tuple(system.residual_names)
        self.input_names = tuple(system.model.input_spec)
        self.output_names = tuple(system.model.output_spec)

    @property
    def u_size(self) -> int:
        return self.ulayout.storage_size()

    @property
    def g_size(self) -> int:
        return self.glayout.storage_size()

    @property
    def b_size(self) -> int:
        return self.blayout.storage_size()

    def _disassemble_flat(
        self,
        layout: AxisLayout,
        tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        offset = 0
        for group in layout.groups:
            for name in group:
                type_cls = layout.type_of(name)
                size = layout.var_size(name)
                values[name] = _unflatten_base(tensor[..., offset : offset + size], type_cls)
                offset += size
        return values

    def _identity_seed(
        self,
        state: dict[str, torch.Tensor],
        names: tuple[str, ...],
    ) -> ChainRuleDict:
        seed: ChainRuleDict = {}
        for name in names:
            type_cls = self.model.input_spec[name]
            value = state[name]
            dyn_shape = tuple(_batch_shape(value, type_cls))
            seed[name] = {
                name: _expanded_identity_seed(
                    type_cls,
                    (),
                    dyn_shape,
                    dtype=value.dtype,
                    device=value.device,
                )
            }
        return seed

    def _call_model(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
        seed_names: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], ChainRuleDict]:
        state = {
            **self._disassemble_flat(self.glayout, g_flat),
            **self._disassemble_flat(self.ulayout, u_flat),
        }
        seed = self._identity_seed(state, seed_names) if seed_names else None
        # wrap each raw state tensor as its typed wrapper so the
        # downstream ComposedModel preserves the sub_batch_ndim hint. Dense
        # path's unknowns are sub-batch-free so the explicit ``sub_batch_ndim=0``
        # is a no-op; the call exists to satisfy the same wrap contract as
        # ``_BlockSystemModule._call_model`` below.
        args = tuple(self.model.input_spec[name](state[name]) for name in self.input_names)
        result = self.model(*args, v=seed) if seed is not None else self.model(*args)
        result_tuple = result if isinstance(result, tuple) else (result,)
        if seed is None:
            output_values = result_tuple
            v_out: ChainRuleDict = {}
        else:
            output_values = result_tuple[:-1]
            v_out = result_tuple[-1]
        output_state: dict[str, torch.Tensor] = {
            name: value.data if isinstance(value, TensorWrapper) else value
            for name, value in zip(self.output_names, output_values, strict=True)
        }
        return output_state, v_out

    def _flat_residual(self, output_state: dict[str, torch.Tensor]) -> torch.Tensor:
        residual_values = {name: output_state[name] for name in self.residual_names}
        parts = AssembledVector.from_dict(self.blayout, residual_values).tensors
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _zero_block(
        self,
        residual_name: str,
        input_name: str,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        row_size = _storage_size(self.model.output_spec[residual_name])
        col_size = _storage_size(self.model.input_spec[input_name])
        return like.new_zeros(*like.shape[:-1], row_size, col_size)

    def _dense_matrix(
        self,
        col_layout: AxisLayout,
        v_out: ChainRuleDict,
        output_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        residual_values = {name: output_state[name] for name in self.residual_names}
        like_by_row = {
            name: _flatten_base(residual_values[name], self.model.output_spec[name])
            for name in self.residual_names
        }
        row_parts: list[torch.Tensor] = []
        for row_name in self.residual_names:
            col_parts: list[torch.Tensor] = []
            for col_name in col_layout.vars():
                t_block = v_out.get(row_name, {}).get(col_name)
                if t_block is None:
                    block = self._zero_block(row_name, col_name, like=like_by_row[row_name])
                else:
                    block = _tangent_block_to_trailing_k(t_block)
                col_parts.append(block)
            row_parts.append(torch.cat(col_parts, dim=-1))
        return torch.cat(row_parts, dim=-2)


class DenseRHS(_DenseSystemModule):
    """Exportable tensor-only equivalent of `LinearSystem.b()`."""

    def forward(self, u_flat: torch.Tensor, g_flat: torch.Tensor) -> torch.Tensor:
        output_state, _ = self._call_model(u_flat, g_flat, ())
        return -self._flat_residual(output_state)


class DenseOperator(_DenseSystemModule):
    """Exportable tensor-only equivalent of `LinearSystem.A()`."""

    def forward(self, u_flat: torch.Tensor, g_flat: torch.Tensor) -> torch.Tensor:
        output_state, v_out = self._call_model(u_flat, g_flat, self.unknown_names)
        return self._dense_matrix(self.ulayout, v_out, output_state)


class DenseLinearizedSystem(_DenseSystemModule):
    """Exportable tensor-only equivalent of `LinearSystem.A_and_b()`."""

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_state, v_out = self._call_model(u_flat, g_flat, self.unknown_names)
        A = self._dense_matrix(self.ulayout, v_out, output_state)
        b = -self._flat_residual(output_state)
        return A, b


class DenseImplicitSensitivity(_DenseSystemModule):
    """Exportable tensor-only equivalent of `LinearSystem.A_and_B()`."""

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed_names = (*self.unknown_names, *self.given_names)
        output_state, v_out = self._call_model(u_flat, g_flat, seed_names)
        A = self._dense_matrix(self.ulayout, v_out, output_state)
        B = self._dense_matrix(self.glayout, v_out, output_state)
        return A, B


class DenseIFT(_DenseSystemModule):
    """Exportable IFT Jacobian for a converged ImplicitUpdate.

    Forward signature ``(u_flat, g_flat) -> du/dg`` of shape
    ``(*B, u_size, g_size)``. The math is the implicit-function theorem at the
    converged state ``u*(g)``:

    .. math::

        r(u^*(g), g) = 0
        \\Rightarrow A \\cdot \\frac{du^*}{dg} = -B
        \\Rightarrow \\frac{du^*}{dg} = -A^{-1} B

    where $A = ∂r/∂u$ and $B = ∂r/∂g$ evaluated at $u = u*(g)$. Both come
    from the same identity-seeded chain-rule pass that
    :class:`DenseImplicitSensitivity` performs; we just bake the linear solve
    into the exported graph so the C++ orchestrator can read the full du/dg
    matrix in one call (no per-iter solve, no Python interaction).

    Bundled into the implicit segment's ``<name>_ift.pt2`` and consumed by
    :class:`AOTIModel` when ``set_value(dout=true)`` chains the master
    Jacobian across the breakpoint.
    """

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> torch.Tensor:
        seed_names = (*self.unknown_names, *self.given_names)
        output_state, v_out = self._call_model(u_flat, g_flat, seed_names)
        A = self._dense_matrix(self.ulayout, v_out, output_state)
        B = self._dense_matrix(self.glayout, v_out, output_state)
        # du/dg = -A^-1 B; solve once for all g columns.
        return -torch.linalg.solve(A, B)


class DenseNewtonStep(_DenseSystemModule):
    """Exportable single Newton iteration: assemble + solve + update + new RHS.

    Forward signature ``(u_flat, g_flat) -> (u_new_flat, b_new_flat)``:

    1. Assemble $A = dr/du$ and ``b = -r`` at the current $u$.
    2. Solve $A * du = b$ via :func:`torch.linalg.solve` (batched 8x8 LU
       fused into the exported graph).
    3. Update $u_new = u + du$.
    4. Re-evaluate the residual at $u_new$ and return ``b_new = -r(u_new)``
       so the caller can perform its convergence check without another
       AOTI roundtrip.

    The C++ Newton loop using a `.pt2` of this class becomes essentially

    .. code-block:: cpp

        for (i = 1; i < miters; ++i) {
            std::tie(u, b) = step_pt2.run({u, g});
            if (converged(b, b0)) return;
        }

    with no per-iter C++ linear-solve or in-place update. Inductor sees the
    whole iteration as one graph and is free to fuse aggressively.
    """

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Step 1+2: assemble A, b at u and solve A * du = b ----
        output_state, v_out = self._call_model(u_flat, g_flat, self.unknown_names)
        A = self._dense_matrix(self.ulayout, v_out, output_state)
        b = -self._flat_residual(output_state)
        du = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

        # ---- Step 3: take the full Newton step ----
        u_new_flat = u_flat + du

        # ---- Step 4: residual at u_new for the convergence check ----
        output_state_new, _ = self._call_model(u_new_flat, g_flat, ())
        b_new = -self._flat_residual(output_state_new)

        return u_new_flat, b_new


# =============================================================================
# Block* export wrappers — multi-group BLOCK+DENSE Schur path
# =============================================================================


class _BlockSystemModule(nn.Module):
    """Tensor-only export surface for a frozen ``ModelNonlinearSystem`` whose
    linear solver is :class:`~solvers.SchurComplement`.

    Mirrors :class:`_DenseSystemModule` but threads the Phase-10.8 sub-batch
    metadata (per-variable ``sub_batch_shapes``, inferred ``dynamic_batch_ndim``)
    so the assembled blocks land on the per-group flat layout that
    ``SchurComplement.solve`` expects. Each per-group flat tensor packs every
    variable's full ``(*dyn, *sub_batch, *base)`` storage into one
    $(*dyn, sum_{v in group}(prod(sub_batch_v) * base_size_v))$ axis;
    concatenating the per-group flats gives the single ``(*dyn, u_size)``
    surface tensor that ``DenseRHS`` and ``DenseNewtonStep`` use, so C++
    AOTI orchestration sees one uniform ``(u_flat, g_flat) → ...``
    contract regardless of whether the inside is Dense or Block.
    """

    def __init__(self, system: ModelNonlinearSystem) -> None:
        super().__init__()
        self.model = system.model
        self.ulayout = system.ulayout
        self.glayout = system.glayout
        self.blayout = system.blayout
        self.unknown_names = tuple(system.unknown_names)
        self.given_names = tuple(system.given_names)
        self.residual_names = tuple(system.residual_names)
        self.input_names = tuple(system.model.input_spec)
        self.output_names = tuple(system.model.output_spec)
        self.dyn_ndim: dict[str, int] = dict(system._dynamic_batch_ndim)
        self.sub_batch_shapes = dict(system._sub_batch_shapes)

    # ---- size accessors ----

    @property
    def u_size(self) -> int:
        return _layout_total_storage(self.ulayout)

    @property
    def g_size(self) -> int:
        return _layout_total_storage(self.glayout)

    @property
    def b_size(self) -> int:
        return _layout_total_storage(self.blayout)

    # ---- packing / unpacking ----

    def _per_var_storage(self, layout: AxisLayout, name: str) -> int:
        sb = layout.sub_batch_shape(name)
        sb_total = 1
        for s in sb:
            sb_total *= int(s)
        return sb_total * layout.var_size(name)

    def _disassemble_flat(
        self,
        layout: AxisLayout,
        tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Split ``(*dyn, total)`` flat tensor into per-variable
        ``(*dyn, *sub_batch, *base)`` storages (D-052 inverse of pack)."""
        values: dict[str, torch.Tensor] = {}
        offset = 0
        for group in layout.groups:
            for name in group:
                type_cls = layout.type_of(name)
                sb = layout.sub_batch_shape(name)
                size = self._per_var_storage(layout, name)
                slab = tensor[..., offset : offset + size]
                values[name] = _unflatten_sub_batch_and_base(
                    slab, type_cls, sb, self.dyn_ndim[name]
                )
                offset += size
        return values

    def _flat_residual(self, output_state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pack residual values into a single ``(*dyn, b_size)`` flat tensor."""
        residual_values = {name: output_state[name] for name in self.residual_names}
        per_group = AssembledVector.from_dict(self.blayout, residual_values).tensors
        return torch.cat(per_group, dim=-1) if len(per_group) > 1 else per_group[0]

    # ---- call the model + identity seeding ----

    def _call_model(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
        seed_names: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], ChainRuleDict]:
        state = {
            **self._disassemble_flat(self.glayout, g_flat),
            **self._disassemble_flat(self.ulayout, u_flat),
        }
        seed: ChainRuleDict | None
        if seed_names:
            seed = {}
            for name in seed_names:
                type_cls = self.model.input_spec[name]
                value = state[name]
                sub_batch_shape = tuple(self.sub_batch_shapes.get(name, ()))
                dyn_shape = tuple(_batch_shape(value, type_cls)[: self.dyn_ndim[name]])
                seed[name] = {
                    name: _expanded_identity_seed(
                        type_cls,
                        sub_batch_shape,
                        dyn_shape,
                        dtype=value.dtype,
                        device=value.device,
                    )
                }
        else:
            seed = None
        # Wrap each raw state tensor with its system-inferred sub_batch_ndim
        # so the downstream ComposedModel preserves the hint.
        args = tuple(
            self.model.input_spec[name](
                state[name],
                sub_batch_ndim=len(self.sub_batch_shapes.get(name, ())),
            )
            for name in self.input_names
        )
        result = self.model(*args, v=seed) if seed is not None else self.model(*args)
        result_tuple = result if isinstance(result, tuple) else (result,)
        if seed is None:
            output_values = result_tuple
            v_out: ChainRuleDict = {}
        else:
            output_values = result_tuple[:-1]
            v_out = result_tuple[-1]
        output_state: dict[str, torch.Tensor] = {
            name: value.data if isinstance(value, TensorWrapper) else value
            for name, value in zip(self.output_names, output_values, strict=True)
        }
        return output_state, v_out

    # ---- multi-group A / B assembly + per-group flat residual ----

    def _assembled_matrix(
        self,
        col_layout: AxisLayout,
        v_out: ChainRuleDict,
        output_state: dict[str, torch.Tensor],
    ) -> AssembledMatrix:
        """Multi-group AssembledMatrix via the Phase-10.8 sub-batch-aware
        ``_assemble_matrix`` logic (per-row sub-batch flattened into the row dim;
        each (i, j) block has shape ``(*dyn, R_total_i, C_total_j)``)."""
        residual_values = {name: output_state[name] for name in self.residual_names}
        like_by_row = {
            name: _flatten_base(residual_values[name], self.model.output_spec[name])
            for name in self.residual_names
        }
        return _build_block_matrix(
            self.model,
            self.blayout,
            col_layout,
            v_out,
            like_by_row,
            self.dyn_ndim,
        )

    def _assembled_b(self, output_state: dict[str, torch.Tensor]) -> AssembledVector:
        """Multi-group b = -r as ``AssembledVector`` (per-group tensors)."""
        residual_values = {name: output_state[name] for name in self.residual_names}
        b = AssembledVector.from_dict(self.blayout, residual_values)
        return AssembledVector(b.layout, [-t for t in b.tensors])


def _layout_total_storage(layout: AxisLayout) -> int:
    """Total flat storage across all groups, accounting for sub-batch."""
    total = 0
    for group in layout.groups:
        for name in group:
            sb = layout.sub_batch_shape(name)
            sb_total = 1
            for s in sb:
                sb_total *= int(s)
            total += sb_total * layout.var_size(name)
    return total


def _build_block_matrix(
    model: Model,
    row_layout: AxisLayout,
    col_layout: AxisLayout,
    v_out: ChainRuleDict,
    like_by_row: dict[str, torch.Tensor],
    dyn_ndim: dict[str, int],
) -> AssembledMatrix:
    """Stateless mirror of ``ModelNonlinearSystem._assemble_matrix``.

    Extracted so ``_BlockSystemModule`` can build per-group blocks during
    ``torch.export`` tracing without depending on a live ``ModelNonlinearSystem``
    instance. ``dyn_ndim`` is the per-row-variable dynamic-batch ndim
    (deterministic, as declared by the caller).
    """
    col_K: dict[str, int] = {}
    for group in col_layout.groups:
        for name in group:
            sb = col_layout.sub_batch_shape(name)
            sb_total = 1
            for s in sb:
                sb_total *= int(s)
            col_K[name] = sb_total * col_layout.var_size(name)

    rows: list[list[torch.Tensor]] = []
    for row_group in row_layout.groups:
        block_row: list[torch.Tensor] = []
        for col_group in col_layout.groups:
            row_parts: list[torch.Tensor] = []
            for row_name in row_group:
                row_sb = row_layout.sub_batch_shape(row_name)
                # Infer dyn from the ACTUAL residual value's shape, not from
                # the per-unknown ``dyn_ndim`` dict. See companion comment in
                # ``ModelNonlinearSystem._assemble_matrix``.
                row_total_ndim = like_by_row[row_name].ndim - 1
                row_dyn_ndim = row_total_ndim - len(row_sb)
                row_batch_dyn = like_by_row[row_name].shape[:row_dyn_ndim]
                row_batch_full = tuple(row_batch_dyn) + tuple(row_sb)
                _ = dyn_ndim  # parameter kept for backward-compatible signature
                col_parts: list[torch.Tensor] = []
                for col_name in col_group:
                    t_block = v_out.get(row_name, {}).get(col_name)
                    if t_block is None:
                        row_size = _storage_size(model.output_spec[row_name])
                        block = like_by_row[row_name].new_zeros(
                            *row_batch_full, row_size, col_K[col_name]
                        )
                    else:
                        block = _broadcast_block_to_row_batch(
                            _tangent_block_to_trailing_k(t_block),
                            row_batch_full,
                        )
                    col_parts.append(block)
                cat_cols = torch.cat(col_parts, dim=-1)
                K_total = cat_cols.shape[-1]
                cat_cols = cat_cols.reshape(*row_batch_dyn, -1, K_total)
                row_parts.append(cat_cols)
            # Broadcast row_parts to a common leading-batch shape before
            # catting at the row dim. See companion comment in
            # ``ModelNonlinearSystem._assemble_matrix``.
            if len(row_parts) > 1:
                lead = torch.broadcast_shapes(*[p.shape[:-2] for p in row_parts])
                row_parts = [p.broadcast_to((*lead, *p.shape[-2:])) for p in row_parts]
            block_row.append(torch.cat(row_parts, dim=-2))
        rows.append(block_row)
    return AssembledMatrix(row_layout, col_layout, rows)


class BlockRHS(_BlockSystemModule):
    """Exportable tensor-only equivalent of ``LinearSystem.b()`` for the
    multi-group Schur path. Same ``(u_flat, g_flat) → b_flat`` contract as
    :class:`DenseRHS` — block-ness is internal."""

    def forward(self, u_flat: torch.Tensor, g_flat: torch.Tensor) -> torch.Tensor:
        output_state, _ = self._call_model(u_flat, g_flat, ())
        return -self._flat_residual(output_state)


class BlockNewtonStep(_BlockSystemModule):
    """Exportable single Newton iteration with mixed BLOCK+DENSE Schur.

    Same ``(u_flat, g_flat) → (u_new_flat, b_new_flat)`` contract as
    :class:`DenseNewtonStep`. Inside:

    1. Disassemble ``u_flat`` / ``g_flat`` into per-variable
       ``(*dyn, *sub_batch, *base)`` storage.
    2. Call the residual model with per-sub-batch-site identity seeds.
    3. Assemble multi-group ``A: AssembledMatrix`` and ``b: AssembledVector``.
    4. Solve ``A du = b`` via :class:`~solvers.SchurComplement` (six-step block
       factorisation; $A_pp^-1$ from ``DenseLU``, Schur complement from
       ``S = A_ss - A_sp @ Y``).
    5. Re-flatten ``du`` to ``du_flat`` and form $u_new = u + du$.
    6. Re-evaluate residual at $u_new$ and return ``-r(u_new)`` so the C++
       Newton loop can perform its convergence check without another roundtrip.

    The C++ ``aoti::Model::_solve_newton`` orchestrator sees the same
    input/output signature as the Dense path and requires no per-solver
    dispatch.
    """

    def __init__(
        self,
        system: ModelNonlinearSystem,
        schur_solver,
    ) -> None:
        super().__init__(system)
        # SchurComplement instance held as a Python attribute (not a submodule
        # — it carries no parameters). Holds the primary/Schur DenseLU
        # sub-solvers and the group indices.
        self._schur = schur_solver

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Step 1+2: disassemble inputs, seed unknowns, assemble system ----
        output_state, v_out = self._call_model(u_flat, g_flat, self.unknown_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        b_assembled = self._assembled_b(output_state)

        # ---- Step 3: Schur-factorised linear solve ----
        du = self._schur.solve(A, b_assembled)
        # ``du`` is an AssembledVector with one tensor per group; flatten to
        # the single (*dyn, u_size) surface contract.
        du_flat = torch.cat(du.tensors, dim=-1) if len(du.tensors) > 1 else du.tensors[0]

        # ---- Step 4: full Newton step ----
        u_new_flat = u_flat + du_flat

        # ---- Step 5: residual at u_new for the C++ convergence check ----
        output_state_new, _ = self._call_model(u_new_flat, g_flat, ())
        b_new = -self._flat_residual(output_state_new)
        return u_new_flat, b_new


class BlockIFT(_BlockSystemModule):
    """Exportable IFT Jacobian $du/dg = -A^-1 B$ for a converged
    Schur-based ImplicitUpdate. Same ``(u_flat, g_flat) → du/dg`` contract as
    :class:`DenseIFT`; the per-group blocks of $B$ are solved via
    :class:`~solvers.SchurComplement`'s matrix-RHS overload."""

    def __init__(
        self,
        system: ModelNonlinearSystem,
        schur_solver,
    ) -> None:
        super().__init__(system)
        self._schur = schur_solver

    def forward(
        self,
        u_flat: torch.Tensor,
        g_flat: torch.Tensor,
    ) -> torch.Tensor:
        seed_names = (*self.unknown_names, *self.given_names)
        output_state, v_out = self._call_model(u_flat, g_flat, seed_names)
        A = self._assembled_matrix(self.ulayout, v_out, output_state)
        B = self._assembled_matrix(self.glayout, v_out, output_state)
        du_dg = self._schur.solve(A, B)
        # Negate and re-flatten per-group blocks into a single (*dyn, u, g) matrix.
        neg = [[-t for t in row] for row in du_dg.tensors]
        row_concat = [torch.cat(row, dim=-1) for row in neg]
        return torch.cat(row_concat, dim=-2)


__all__ = [
    "AxisLayout",
    "AssembledVector",
    "AssembledMatrix",
    "LinearSystem",
    "NonlinearSystem",
    "ModelNonlinearSystem",
    "DenseRHS",
    "DenseOperator",
    "DenseLinearizedSystem",
    "DenseImplicitSensitivity",
    "DenseIFT",
    "DenseNewtonStep",
    "BlockRHS",
    "BlockNewtonStep",
    "BlockIFT",
    "norm",
]
