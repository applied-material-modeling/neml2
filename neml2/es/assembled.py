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

"""Assembled vector / matrix blocks on top of :class:`~neml2.types.Tensor`.

v2-parity (V2P-6 rewrite). Each variable group declares a
:data:`~neml2.es.axis_layout.SubBatchStructure` -- either ``"block"`` or
``"dense"`` -- on the owning :class:`AxisLayout`. Assembly behaviour
branches on this flag:

* **Block group**: the group's sub_batch axes are PRESERVED as intermediate
  dims on the per-group assembled tensor.

  * Vector: ``(*dyn, *sub_batch, group_storage_size)`` -- per-site
    independence preserved; the underlying solver batches per site
    naturally.
  * Matrix block: ``(*dyn, *intmd_row, *intmd_col, row_storage, col_storage)``.
    Each side's sub_batch axes appear as intmd dims (in row order then
    col order). For a per-grain × per-grain block both sides share the
    same per-grain extent and the intmd dims double-stack with the
    eye-diagonal storage convention.

* **Dense group**: the group's sub_batch axes are FOLDED into the base
  storage at assembly time.

  * Vector: ``(*dyn, group_storage_size_with_sub_folded)``.
  * Matrix block: ``(*dyn, row_storage, col_storage)``.

Per-block decision in ``_build_group_block``:

* both row group AND col group dense -> dense block (fold everything).
* either side block -> keep that side's sub_batch as intmd on the block.

``AssembledMatrix.__matmul__`` mirrors v2: standard ``mm`` reduces along
the base axes; an additional ``sum`` over the intmd axes fires when the
contracting (inner) group's :data:`SubBatchStructure` is ``"block"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, overload

import torch

from neml2.types import Tensor, TensorWrapper
from neml2.types import cat as _cat

from ._helpers import _storage_size, _tangent_block_to_trailing_k
from .axis_layout import AxisLayout

if TYPE_CHECKING:
    from neml2.models.chain_rule import ChainRuleDict
    from neml2.models.model import Model

    from .sparse import SparseMatrix, SparseVector


# ---------------------------------------------------------------------------
# AssembledVector
# ---------------------------------------------------------------------------


@dataclass
class AssembledVector:
    """Per-group dense vector blocks.

    Each group's tensor follows the :data:`SubBatchStructure` of the owning
    :class:`AxisLayout`:

    * ``"block"`` -> ``(*dyn, *sub_batch, group_storage_size)``,
      ``base_ndim=1``, ``sub_batch_ndim=len(sub_batch_shape)``.
    * ``"dense"`` -> ``(*dyn, group_storage_size_with_sub_folded)``,
      ``base_ndim=1``, ``sub_batch_ndim=0``.

    Arithmetic and dot products forward to :class:`Tensor` ops.
    """

    layout: AxisLayout
    tensors: list[Tensor]

    @classmethod
    def from_dict(
        cls,
        layout: AxisLayout,
        values: Mapping[str, TensorWrapper],
    ) -> AssembledVector:
        """Pack typed-wrapper ``values`` (one per variable) into per-group tensors.

        For a ``"block"`` group, the per-variable wrappers are flattened to
        base then concatenated along base (last) axis -- sub_batch axes
        stay as intermediate dims on the resulting Tensor.

        For a ``"dense"`` group, each wrapper's sub_batch is folded INTO
        the variable's flat base storage before concat: per-variable
        contribution becomes ``(*dyn, sub_total * base_size)``, then all
        contributions concat to ``(*dyn, sum_var (sub_total * base_size))``.

        Per CLAUDE.md rule 1: ``values`` is strictly typed -- raw
        ``torch.Tensor`` is rejected. External boundaries wrap with the
        appropriate ``TensorWrapper`` subclass at the construction site.
        """
        tensors: list[Tensor] = []
        for gi, group in enumerate(layout.groups):
            structure = layout.structure[gi]
            parts: list[Tensor] = []
            for name in group:
                wrapped = values[name]
                t = Tensor.from_typed(wrapped).flatten_base()
                if structure == "dense" and t.sub_batch_ndim > 0:
                    t = t.flatten_sub_batch_into_first_base_axis()
                parts.append(t)
            tensors.append(_cat_along_base(parts))
        return cls(layout, tensors)

    def disassemble(self) -> SparseVector:
        """Unpack per-group tensors back into a :class:`SparseVector`.

        Inverse of :meth:`from_dict` -- the per-group split point is the
        per-variable storage size; for ``"dense"`` groups the sub_batch
        is unfolded back to the declared shape before re-typing.

        Returns a :class:`SparseVector` (the typed dual of this object).
        The underlying name -> typed-wrapper dict is on
        ``.values`` for callers that need the raw mapping.
        """
        from .sparse import SparseVector  # local: avoid circular import

        out: dict[str, TensorWrapper] = {}
        for gi, group in enumerate(self.layout.groups):
            structure = self.layout.structure[gi]
            t = self.tensors[gi]
            slices = self._split_group(t, group, structure)
            for name, part in zip(group, slices, strict=True):
                type_cls = self.layout.specs[name]
                sb = self.layout.sub_batch_shape(name)
                if structure == "dense" and sb:
                    # Unfold sub_batch from the trailing base axis.
                    var_base = _storage_size(type_cls)
                    # part has shape (*dyn, sub_total * var_base); reshape to
                    # (*dyn, *sub, var_base) then unflatten base.
                    target = (*part.data.shape[:-1], *tuple(int(s) for s in sb), var_base)
                    raw = part.data.reshape(target)
                    new_t = Tensor(
                        raw,
                        batch_ndim=part.batch_ndim,
                        sub_batch_ndim=len(sb),
                    )
                    out[name] = new_t.unflatten_base(*type_cls.BASE_SHAPE).as_typed(type_cls)
                else:
                    out[name] = part.unflatten_base(*type_cls.BASE_SHAPE).as_typed(type_cls)
        return SparseVector(self.layout, out)

    def _split_group(
        self,
        t: Tensor,
        group: tuple[str, ...],
        structure: str,
    ) -> list[Tensor]:
        per_var_sizes = [
            (
                _storage_size(self.layout.specs[n])
                * (prod([int(s) for s in self.layout.sub_batch_shape(n)]) or 1)
                if structure == "dense"
                else _storage_size(self.layout.specs[n])
            )
            for n in group
        ]
        cuts: list[Tensor] = []
        start = 0
        flat = t.data
        for size in per_var_sizes:
            slc = flat[..., start : start + size]
            cuts.append(Tensor(slc, batch_ndim=t.batch_ndim, sub_batch_ndim=t.sub_batch_ndim))
            start += size
        return cuts

    def __neg__(self) -> AssembledVector:
        return AssembledVector(self.layout, [-t for t in self.tensors])

    def __add__(self, other: AssembledVector) -> AssembledVector:
        if self.layout != other.layout:
            raise ValueError("AssembledVector + requires matching layouts")
        return AssembledVector(
            self.layout, [a + b for a, b in zip(self.tensors, other.tensors, strict=True)]
        )

    def __sub__(self, other: AssembledVector) -> AssembledVector:
        if self.layout != other.layout:
            raise ValueError("AssembledVector - requires matching layouts")
        return AssembledVector(
            self.layout, [a - b for a, b in zip(self.tensors, other.tensors, strict=True)]
        )

    def group(self, index: int) -> AssembledVector:
        return AssembledVector(self.layout.sub_layout(index), [self.tensors[index]])


def wrap_group_raw(
    raw: torch.Tensor,
    group_names: tuple[str, ...],
    structure: str,
    layout: AxisLayout,
) -> Tensor:
    """Wrap a per-group raw tensor into a typed dynamic-base :class:`Tensor`.

    Inverse of the per-group ``.data`` extraction at the solver / AOTI
    framework boundary: it infers ``batch_ndim`` / ``sub_batch_ndim`` from the
    group's declared :data:`SubBatchStructure` so the result drops straight
    into an :class:`AssembledVector` and round-trips through
    :meth:`AssembledVector.disassemble`. Used wherever per-group raw tensors
    re-enter the typed world (the eager Newton boundary, the AOTI export
    segment inputs).
    """
    if not group_names:
        return Tensor(raw, batch_ndim=raw.ndim, sub_batch_ndim=0)
    first_name = group_names[0]
    if structure == "block":
        # (*dyn, *sub_batch, group_base_total): base_ndim=1.
        sub_ndim = len(layout.sub_batch_shape(first_name))
        return Tensor(raw, batch_ndim=raw.ndim - sub_ndim - 1, sub_batch_ndim=sub_ndim)
    # DENSE: (*dyn, group_total); sub_batch folded into base.
    return Tensor(raw, batch_ndim=raw.ndim - 1, sub_batch_ndim=0)


def group_block_sub_batch_ndim(
    row_layout: AxisLayout,
    row_gi: int,
    col_layout: AxisLayout,
    col_gi: int,
) -> int:
    """The ``sub_batch_ndim`` of the assembled ``(row_gi, col_gi)`` block.

    Single source of truth mirroring the storage convention in
    :func:`_build_group_block` / :func:`_zero_block` /
    :func:`_convert_tangent_to_paired_block`: a DENSE side folds its sub_batch
    into base (0 intmd axes); a BLOCK side keeps its sub_batch as intmd axes; a
    BLOCK+BLOCK pair with identical per-site shape is stored on the compact
    paired diagonal (a single per-site axis, not the full grid).

    Used to reconstruct a typed :class:`AssembledMatrix` from raw per-block
    tensors at the solver graph boundary (:class:`~neml2.es.implicit.LinearSolve`):
    only the (structural) ``sub_batch_ndim`` is needed there -- ``batch_ndim``
    follows from the runtime tensor's ndim.
    """
    rs = row_layout.structure[row_gi]
    cs = col_layout.structure[col_gi]
    r_sub = tuple(int(s) for s in row_layout.group_sub_batch_shape(row_gi))
    c_sub = tuple(int(s) for s in col_layout.group_sub_batch_shape(col_gi))
    if rs == "block" and cs == "block" and r_sub == c_sub and len(r_sub) > 0:
        return len(r_sub)  # paired diagonal (compact)
    r_intmd = len(r_sub) if rs == "block" else 0
    c_intmd = len(c_sub) if cs == "block" else 0
    return r_intmd + c_intmd


def wrap_block_raw(raw: torch.Tensor, sub_batch_ndim: int) -> Tensor:
    """Wrap a raw assembled-matrix block into a typed dynamic-base :class:`Tensor`.

    Inverse of the per-block ``.data`` extraction at the solver graph boundary: a
    block is ``(*dyn, *sub, row_storage, col_storage)`` (``base_ndim=2``), so
    ``batch_ndim = raw.ndim - sub_batch_ndim - 2``. ``sub_batch_ndim`` comes from
    :func:`group_block_sub_batch_ndim`.
    """
    return Tensor(raw, batch_ndim=raw.ndim - sub_batch_ndim - 2, sub_batch_ndim=sub_batch_ndim)


# ---------------------------------------------------------------------------
# AssembledMatrix
# ---------------------------------------------------------------------------


@dataclass
class AssembledMatrix:
    """2D grid of per-(row_group, col_group) tensor blocks.

    Per-block storage decision:

    * both row group AND col group dense -> shape
      ``(*dyn, row_storage_with_sub_folded, col_storage_with_sub_folded)``;
      ``base_ndim=2``, ``sub_batch_ndim=0``.
    * any side block -> that side's sub_batch is preserved as intmd dims.
      Shape ``(*dyn, *intmd_row, *intmd_col, row_storage, col_storage)``;
      ``base_ndim=2``, ``sub_batch_ndim = len(intmd_row) + len(intmd_col)``.

    matmul:

    * ``mm(aik, bkj)`` contracts the trailing two base axes.
    * if ``col_layout.structure[k] == "block"`` (the inner group is block),
      a sub_batch reduction (``sum`` over the intmd dims contributed by
      group k) runs after the per-block mm. This is the v2-parity
      "block intmd_sum after mm" pattern.
    """

    row_layout: AxisLayout
    col_layout: AxisLayout
    tensors: list[list[Tensor]] = field(default_factory=list)

    def __neg__(self) -> AssembledMatrix:
        return AssembledMatrix(
            self.row_layout,
            self.col_layout,
            [[-t for t in row] for row in self.tensors],
        )

    def __add__(self, other: AssembledMatrix) -> AssembledMatrix:
        if self.row_layout != other.row_layout or self.col_layout != other.col_layout:
            raise ValueError("AssembledMatrix + requires matching layouts")
        return AssembledMatrix(
            self.row_layout,
            self.col_layout,
            [
                [a + b for a, b in zip(ra, rb, strict=True)]
                for ra, rb in zip(self.tensors, other.tensors, strict=True)
            ],
        )

    def __sub__(self, other: AssembledMatrix) -> AssembledMatrix:
        if self.row_layout != other.row_layout or self.col_layout != other.col_layout:
            raise ValueError("AssembledMatrix - requires matching layouts")
        return AssembledMatrix(
            self.row_layout,
            self.col_layout,
            [
                [a - b for a, b in zip(ra, rb, strict=True)]
                for ra, rb in zip(self.tensors, other.tensors, strict=True)
            ],
        )

    def group(self, i: int, j: int) -> AssembledMatrix:
        return AssembledMatrix(
            self.row_layout.sub_layout(i),
            self.col_layout.sub_layout(j),
            [[self.tensors[i][j]]],
        )

    def disassemble(self) -> SparseMatrix:
        """Unpack per-block tensors into a :class:`SparseMatrix`.

        Each cell ``cells[row_var][col_var]`` is a typed dynamic-base
        :class:`~neml2.types.Tensor` that carries the block's ``batch_ndim``
        / ``sub_batch_ndim`` verbatim. For ``"dense"`` blocks the sub_batch
        axes have been folded into base, so the slices are flat in row +
        col. Boundary callers (e.g. the pyzag interface) read the
        underlying dict-of-dict via ``.cells`` and unwrap to raw ``.data``
        at their own framework boundary.
        """
        from .sparse import SparseMatrix  # local: avoid circular import

        out: dict[str, dict[str, Tensor]] = {}
        for row_group in self.row_layout.groups:
            for name in row_group:
                out[name] = {}
        for gi, row_group in enumerate(self.row_layout.groups):
            row_structure = self.row_layout.structure[gi]
            row_sizes = [
                self._var_storage(self.row_layout, gi, n, row_structure) for n in row_group
            ]
            for gj, col_group in enumerate(self.col_layout.groups):
                col_structure = self.col_layout.structure[gj]
                col_sizes = [
                    self._var_storage(self.col_layout, gj, n, col_structure) for n in col_group
                ]
                t = self.tensors[gi][gj]
                # Split last (col) then -2 (row), keeping the typed Tensor
                # wrapper so batch_ndim / sub_batch_ndim survive at the
                # per-block surface (no .data leakage).
                r_start = 0
                for ri, rname in enumerate(row_group):
                    r_size = row_sizes[ri]
                    c_start = 0
                    for cj, cname in enumerate(col_group):
                        c_size = col_sizes[cj]
                        out[rname][cname] = t.base[
                            ..., r_start : r_start + r_size, c_start : c_start + c_size
                        ]
                        c_start += c_size
                    r_start += r_size
        return SparseMatrix(self.row_layout, self.col_layout, out)

    @staticmethod
    def select_blocks(
        row_layout: AxisLayout,
        col_layout: AxisLayout,
        blocks: dict[str, dict[str, Tensor]],
    ) -> AssembledMatrix:
        """Build an :class:`AssembledMatrix` from per-(row_var, col_var) typed blocks.

        Inverse of :meth:`disassemble`. Each ``blocks[row_name][col_name]`` is
        a typed dynamic-base :class:`~neml2.types.Tensor` with
        ``base_ndim=2`` and ``base_shape == (row_storage, col_storage)``
        sized to match :meth:`_var_storage` for the given layout structure.
        Missing (row_var, col_var) pairs are zero-filled.

        Only the dense-x-dense case is supported (no intmd sub_batch dims on
        either side) -- that's the only path the pyzag interface and the
        round-trip test currently exercise. Block-structure layouts would need
        the corresponding sub_batch axes to live in each block's storage
        and the cat axes shifted accordingly; raise so callers don't
        silently get the wrong shape.
        """
        for structure in (*row_layout.structure, *col_layout.structure):
            if structure != "dense":
                raise NotImplementedError(
                    f"select_blocks only handles dense layouts; got structure={structure!r}"
                )
        # Find one present block to source the dyn-batch shape, dtype, and
        # device for the zero-fill on missing (row_var, col_var) pairs.
        ref: Tensor | None = None
        for cols in blocks.values():
            for t in cols.values():
                ref = t
                break
            if ref is not None:
                break
        if ref is None:
            raise ValueError("select_blocks: empty blocks dict (need at least one to size zeros)")
        batch_ndim = ref.batch_ndim
        dyn_shape = tuple(ref.batch_shape)
        dtype = ref.dtype
        device = ref.device
        out: list[list[Tensor]] = []
        for gi, row_group in enumerate(row_layout.groups):
            row_structure = row_layout.structure[gi]
            row_sizes = [
                AssembledMatrix._var_storage(row_layout, gi, n, row_structure) for n in row_group
            ]
            row_blocks: list[Tensor] = []
            for gj, col_group in enumerate(col_layout.groups):
                col_structure = col_layout.structure[gj]
                col_sizes = [
                    AssembledMatrix._var_storage(col_layout, gj, n, col_structure)
                    for n in col_group
                ]
                # Assemble a (sum(row_sizes), sum(col_sizes)) block by laying
                # the per-(rname, cname) entries out in row-major order via
                # typed base-region cat.
                row_strips: list[Tensor] = []
                for ri, rname in enumerate(row_group):
                    r_size = row_sizes[ri]
                    col_strips: list[Tensor] = []
                    for cj, cname in enumerate(col_group):
                        c_size = col_sizes[cj]
                        b = blocks.get(rname, {}).get(cname)
                        if b is None:
                            b = Tensor(
                                torch.zeros(*dyn_shape, r_size, c_size, dtype=dtype, device=device),
                                batch_ndim=batch_ndim,
                                sub_batch_ndim=0,
                            )
                        col_strips.append(b)
                    row_strips.append(_cat([t.base for t in col_strips], dim=-1))
                row_blocks.append(_cat([t.base for t in row_strips], dim=-2))
            out.append(row_blocks)
        return AssembledMatrix(row_layout, col_layout, out)

    @staticmethod
    def _var_storage(
        layout: AxisLayout,
        gi: int,
        name: str,
        structure: str,
    ) -> int:
        type_cls = layout.specs[name]
        base = _storage_size(type_cls)
        if structure == "dense":
            sb = layout.sub_batch_shape(name)
            return base * (prod([int(s) for s in sb]) or 1)
        return base

    @overload
    def __matmul__(self, other: AssembledVector) -> AssembledVector: ...
    @overload
    def __matmul__(self, other: AssembledMatrix) -> AssembledMatrix: ...
    def __matmul__(self, other):
        """Block matmul ``C = A @ B`` (matrix RHS) or ``c = A @ b`` (vector RHS).

        v2-parity:
            ``Cij = sum_k mm(Aik, Bkj)`` plus intmd_sum on the result when
            ``col_layout.structure[k] == "block"`` (reduces the BLOCK k's
            intmd axes that survived mm).
        """
        if isinstance(other, AssembledVector):
            if self.col_layout != other.layout:
                raise ValueError(
                    "AssembledMatrix @ AssembledVector requires matching inner layouts"
                )
            out_tensors: list[Tensor] = []
            for i in range(self.row_layout.ngroup):
                acc: Tensor | None = None
                for k in range(self.col_layout.ngroup):
                    aik = self.tensors[i][k]
                    bk = other.tensors[k]
                    r = aik @ bk
                    if self.col_layout.structure[k] == "block":
                        r = _intmd_sum_k(r, self.col_layout, k)
                    acc = r if acc is None else acc + r
                assert acc is not None
                out_tensors.append(acc)
            return AssembledVector(self.row_layout, out_tensors)

        if not isinstance(other, AssembledMatrix):
            raise TypeError(
                f"AssembledMatrix @ expected AssembledVector or AssembledMatrix; "
                f"got {type(other).__name__}"
            )
        if self.col_layout != other.row_layout:
            raise ValueError("AssembledMatrix @ AssembledMatrix requires matching inner layouts")
        out_blocks: list[list[Tensor]] = []
        for i in range(self.row_layout.ngroup):
            row_blocks: list[Tensor] = []
            for j in range(other.col_layout.ngroup):
                acc: Tensor | None = None
                for k in range(self.col_layout.ngroup):
                    aik = self.tensors[i][k]
                    bkj = other.tensors[k][j]
                    r = aik @ bkj
                    if self.col_layout.structure[k] == "block":
                        r = _intmd_sum_k(r, self.col_layout, k)
                    acc = r if acc is None else acc + r
                assert acc is not None
                row_blocks.append(acc)
            out_blocks.append(row_blocks)
        return AssembledMatrix(self.row_layout, other.col_layout, out_blocks)


# ---------------------------------------------------------------------------
# Norms (free functions, mirror v2 names)
# ---------------------------------------------------------------------------


def norm(v: AssembledVector) -> torch.Tensor:
    """Batched Euclidean norm over all assembled vector groups."""
    return torch.sqrt(norm_sq(v))


def norm_sq(v: AssembledVector) -> torch.Tensor:
    """Batched squared Euclidean norm over all assembled vector groups."""
    total: torch.Tensor | None = None
    for gi, t in enumerate(v.tensors):
        # Sum over base axis; for BLOCK groups also sum over intmd sub_batch.
        squared = t.data * t.data
        # Always sum over base axis.
        squared = squared.sum(dim=-1)
        if v.layout.structure[gi] == "block":
            # Sum over the sub_batch intmd axes (they live at positions
            # batch_ndim : batch_ndim + sub_batch_ndim).
            for _ in range(t.sub_batch_ndim):
                squared = squared.sum(dim=-1)
        total = squared if total is None else total + squared
    assert total is not None
    return total


# ---------------------------------------------------------------------------
# Block assembly from chain-rule tangents
# ---------------------------------------------------------------------------


def _build_block_matrix(
    model: Model,
    row_layout: AxisLayout,
    col_layout: AxisLayout,
    v_out: ChainRuleDict,
    like_by_row: Mapping[str, Tensor],
) -> AssembledMatrix:
    """Assemble the full block matrix from chain-rule tangents.

    Iterates ``(row_group, col_group)`` pairs and dispatches to
    :func:`_build_group_block` for each.
    """
    blocks: list[list[Tensor]] = []
    for i, row_group in enumerate(row_layout.groups):
        row_blocks: list[Tensor] = []
        for j, col_group in enumerate(col_layout.groups):
            block = _build_group_block(
                model,
                row_group,
                col_group,
                row_layout,
                col_layout,
                row_structure=row_layout.structure[i],
                col_structure=col_layout.structure[j],
                row_gi=i,
                col_gi=j,
                v_out=v_out,
                like_by_row=like_by_row,
            )
            row_blocks.append(block)
        blocks.append(row_blocks)
    return AssembledMatrix(row_layout, col_layout, blocks)


def _build_group_block(
    model: Model,
    row_group: tuple[str, ...],
    col_group: tuple[str, ...],
    row_layout: AxisLayout,
    col_layout: AxisLayout,
    row_structure: str,
    col_structure: str,
    row_gi: int,
    col_gi: int,
    v_out: ChainRuleDict,
    like_by_row: Mapping[str, Tensor],
) -> Tensor:
    """Assemble one ``(row_group, col_group)`` block as a :class:`Tensor`.

    Per-(row_var, col_var) tangent is converted to trailing-K form, with
    K = (sub_extents × col_base_size) for the col side. The per-variable
    contributions are concatenated along the row axis (-2 in base) and
    column axis (-1 in base).

    Per structure:
    * dense row AND dense col: sub_batch of both sides folded into base
      (rows fold sub_row into row_base, cols' K is already (sub_col_total *
      col_base) from the chain-rule fullify).
    * block row: sub_row stays as leading intmd dim on the assembled block.
    * block col: K still includes sub_col_total * col_base (no separate
      handling needed at assembly; the matmul-side intmd_sum on BLOCK k
      reduces in the trailing kp dim).
    """
    # Determine row dyn shape from the row group's `like` template.
    # All vars in a group share the same dyn shape by construction.
    # ``like`` is the typed dynamic-base Tensor returned by
    # :func:`_flatten_base`, which collapses base to a single trailing
    # axis (BASE_NDIM=0 vars get a unsqueeze; BASE_NDIM>0 vars get
    # reshape). The typed wrapper carries its own ``batch_ndim`` --
    # use that directly instead of computing from total ndim minus
    # BASE_NDIM minus sub, which double-counts for the BASE_NDIM=0
    # case (the trailing axis added by ``_flatten_base`` isn't
    # reflected in the spec's BASE_NDIM).
    first_row = row_group[0]
    like = like_by_row[first_row]
    row_sb = row_layout.sub_batch_shape(first_row)
    row_batch_dyn = tuple(like.shape[: like.batch_ndim])

    # Per-row build.
    row_parts: list[Tensor] = []
    for row_name in row_group:
        col_parts: list[Tensor] = []
        for col_name in col_group:
            col_var_size = _storage_size(col_layout.specs[col_name])
            col_sb = col_layout.sub_batch_shape(col_name)
            col_sb_total = prod([int(s) for s in col_sb]) or 1
            # Expected trailing-K for this col var: sub_col_total * var_base.
            expected_K = col_sb_total * col_var_size
            # Tangent block for this (row, col) pair.
            seed_key = f"{col_name}:rgroup{row_gi}"
            t_block = v_out.get(row_name, {}).get(seed_key)
            if t_block is None:
                # Zero block.
                block = _zero_block(
                    row_batch_dyn,
                    row_sb if row_structure == "block" else (),
                    row_structure,
                    col_structure,
                    row_name,
                    col_name,
                    row_layout,
                    col_layout,
                    like,
                )
            else:
                block = _convert_tangent_to_block(
                    t_block,
                    row_name,
                    col_name,
                    row_layout,
                    col_layout,
                    row_structure,
                    col_structure,
                    row_gi=row_gi,
                    col_gi=col_gi,
                    row_batch_dyn=row_batch_dyn,
                    expected_K=expected_K,
                )
            col_parts.append(block)
        # Concat per-col contributions on trailing base axis (col K).
        row_parts.append(_cat_along_base(col_parts))
    # Concat per-row contributions on row axis (-2).
    if len(row_parts) == 1:
        return row_parts[0]
    return _cat_along_row(row_parts)


def _zero_block(
    row_batch_dyn: tuple[int, ...],
    row_intmd: tuple[int, ...],
    row_structure: str,
    col_structure: str,
    row_name: str,
    col_name: str,
    row_layout: AxisLayout,
    col_layout: AxisLayout,
    like: Tensor,
) -> Tensor:
    row_var_base = _storage_size(row_layout.specs[row_name])
    col_var_base = _storage_size(col_layout.specs[col_name])
    col_sb = col_layout.sub_batch_shape(col_name)
    col_sb_total = prod([int(s) for s in col_sb]) or 1
    row_var_base = _storage_size(row_layout.specs[row_name])
    row_sb = row_layout.sub_batch_shape(row_name)
    row_sb_total = prod([int(s) for s in row_sb]) or 1
    # Paired BLOCK+BLOCK: single per-site intmd axis to match
    # _convert_tangent_to_paired_block's storage convention.
    paired = (
        row_structure == "block"
        and col_structure == "block"
        and tuple(int(s) for s in row_sb) == tuple(int(s) for s in col_sb)
        and len(row_sb) > 0
    )
    if paired:
        site_shape = tuple(int(s) for s in row_sb)
        full_shape = (*row_batch_dyn, *site_shape, row_var_base, col_var_base)
        return Tensor(
            torch.zeros(full_shape, dtype=like.dtype, device=like.device),
            batch_ndim=len(row_batch_dyn),
            sub_batch_ndim=len(site_shape),
        )
    if row_structure == "dense":
        row_dim = row_var_base * row_sb_total
        row_intmd_shape: tuple[int, ...] = ()
    else:
        row_dim = row_var_base
        row_intmd_shape = tuple(int(s) for s in row_sb)
    if col_structure == "dense":
        col_dim = col_var_base * col_sb_total
        col_intmd_shape: tuple[int, ...] = ()
    else:
        col_dim = col_var_base
        col_intmd_shape = tuple(int(s) for s in col_sb)
    full_shape = (
        *row_batch_dyn,
        *row_intmd_shape,
        *col_intmd_shape,
        row_dim,
        col_dim,
    )
    sub_ndim = len(row_intmd_shape) + len(col_intmd_shape)
    return Tensor(
        torch.zeros(full_shape, dtype=like.dtype, device=like.device),
        batch_ndim=len(row_batch_dyn),
        sub_batch_ndim=sub_ndim,
    )


def _convert_tangent_to_paired_block(
    t_block: TensorWrapper,
    paired_sb: tuple[int, ...],
    col_var_base: int,
) -> Tensor:
    """Build the paired-site assembled block ``(*dyn, *site, row_base, col_base)``.

    For a BLOCK+BLOCK pair with identical per-site sub_batch shape on both
    sides, the chain-rule tangent's K_paired axes are broadcast (size 1) in
    storage against per-site sub axes -- the per-site Jacobian is the
    eye-diagonal of the would-be full (site_row × site_col) grid. Storing
    only the diagonal (single paired-site intmd axis) lets the downstream
    linear solver batch over sites as independent per-site inversions.

    Pipeline (no fullify -- we want to keep K_paired compact, not expand):
    1. Materialise any sub_batch broadcast axes so the sub region carries
       the per-site values explicitly.
    2. Squeeze every K_paired axis (size 1 broadcast against the paired
       sub axis) out of leading position.
    3. Move the remaining single K_base axis (size ``col_var_base``) to
       trailing as the col_base storage axis. Flatten the row base axes
       in front of it.
    4. Result shape: ``(*dyn, *site, row_base, col_base)``.
    """
    # Materialise sub_batch broadcast so the per-site sub axes carry their
    # true extent (the chain rule may have left them broadcast against
    # paired K).
    block_w = t_block
    if block_w.sub_batch_state and any(s == "broadcast" for s in block_w.sub_batch_state):
        block_w = block_w.materialize()
    if block_w.sub_batch_ndim != len(paired_sb):
        raise ValueError(
            f"_convert_tangent_to_paired_block: tangent sub_batch_ndim={block_w.sub_batch_ndim} "
            f"doesn't match paired_sb={paired_sb}"
        )
    # Drop broadcast K_paired axes -- they are size 1 in storage and the
    # paired-site information now lives on the sub axis.
    drop_axes = [
        i
        for i, (s, p) in enumerate(zip(block_w.k_state, block_w.k_pairing, strict=True))
        if s == "broadcast" and p is not None and int(block_w.data.shape[i]) == 1
    ]
    data = block_w.data
    new_k_state = list(block_w.k_state)
    new_k_pairing = list(block_w.k_pairing)
    for i in sorted(drop_axes, reverse=True):
        data = data.squeeze(i)
        new_k_state.pop(i)
        new_k_pairing.pop(i)
    new_k_ndim = len(new_k_state)
    if new_k_ndim == 0:
        # Scalar-col or base-trivial case: the chain rule emitted no K_base
        # (col_var_base == 1). Insert a leading size-1 axis so the rest of
        # the pipeline can treat it as the col_base axis.
        if col_var_base != 1:
            raise ValueError(
                f"_convert_tangent_to_paired_block: tangent has no K_base but "
                f"col_var_base={col_var_base} (expected 1 for Scalar cols)"
            )
        data = data.unsqueeze(0)
        new_k_ndim = 1
    elif new_k_ndim != 1:
        raise ValueError(
            f"_convert_tangent_to_paired_block: expected k_ndim=1 after squeezing "
            f"paired-broadcast K, got k_ndim={new_k_ndim} (k_state={new_k_state})"
        )
    if int(data.shape[0]) != col_var_base:
        raise ValueError(
            f"_convert_tangent_to_paired_block: K_base size {data.shape[0]} "
            f"!= col_var_base {col_var_base}"
        )
    base_ndim = type(block_w).BASE_NDIM
    sub_ndim = block_w.sub_batch_ndim
    # Layout after squeeze: (K_base, *dyn, *site, *row_base).
    # Move K_base to trailing -> (*dyn, *site, *row_base, col_base).
    moved = data.movedim(0, -1)
    if base_ndim > 1:
        leading = moved.shape[: -1 - base_ndim]
        site = moved.shape[len(leading) : len(leading) + sub_ndim]
        row_part = moved.shape[len(leading) + sub_ndim : -1]
        row_total = 1
        for s in row_part:
            row_total *= int(s)
        moved = moved.reshape(*leading, *site, row_total, int(moved.shape[-1]))
    elif base_ndim == 0:
        moved = moved.unsqueeze(-2)
    return Tensor(
        moved,
        batch_ndim=moved.ndim - 1 - 1 - sub_ndim,
        sub_batch_ndim=sub_ndim,
    )


def _convert_tangent_to_block(
    t_block: TensorWrapper,
    row_name: str,
    col_name: str,
    row_layout: AxisLayout,
    col_layout: AxisLayout,
    row_structure: str,
    col_structure: str,
    row_gi: int,
    col_gi: int,
    row_batch_dyn: tuple[int, ...],
    expected_K: int,
) -> Tensor:
    """Convert a chain-rule tangent block to the assembled block shape.

    Pipeline:
    1. Move K to trailing via :func:`_tangent_block_to_trailing_k`
       (which fullifies broadcast K_paired axes internally). Result is a
       :class:`Tensor` with shape ``(*dyn, *sub_row, row_base, K)`` and
       ``K = sub_col_total * col_base`` after fullify.
    2. If row_structure == "dense": fold sub_row into row_base. Resulting
       shape: ``(*dyn, sub_row_total * row_base, K)``.
    3. If col_structure == "block": unflatten the trailing K into
       ``(*sub_col, col_base)`` so the col intmd dims sit between row and
       col base. Resulting shape: ``(*dyn, [*sub_row,] *sub_col,
       row_storage, col_base)``.
    """
    row_sb = row_layout.sub_batch_shape(row_name)
    col_sb = col_layout.sub_batch_shape(col_name)
    col_var_base = _storage_size(col_layout.specs[col_name])
    col_sb_total = prod([int(s) for s in col_sb]) or 1

    # Paired BLOCK+BLOCK: same per-site sub_batch shape on both sides means
    # the per-(site_row, site_col) Jacobian is non-zero only on the diagonal
    # site_row == site_col. Store a single intmd axis (paired-site) rather
    # than a sub_row × sub_col grid of mostly-zero cells -- otherwise the
    # downstream torch.linalg.solve sees off-diagonal site cells as
    # independent zero (10, 10) batches and reports singular.
    if (
        row_structure == "block"
        and col_structure == "block"
        and tuple(int(s) for s in row_sb) == tuple(int(s) for s in col_sb)
        and len(row_sb) > 0
    ):
        return _convert_tangent_to_paired_block(
            t_block,
            paired_sb=tuple(int(s) for s in row_sb),
            col_var_base=col_var_base,
        )

    block = _tangent_block_to_trailing_k(t_block)
    # block has trailing-K shape (*dyn, *sub_row, row_base, K)
    # where sub_row = block.sub_batch_shape (may be empty if no per-grain row).
    # Trailing K should equal expected_K after fullify; if smaller (compact
    # tangent never inflated), tile.
    actual_K = int(block.data.shape[-1])
    if actual_K != expected_K:
        if expected_K % actual_K != 0:
            raise ValueError(
                f"chain-rule block K={actual_K} for ({row_name!r},{col_name!r}) "
                f"doesn't divide expected K={expected_K}"
            )
        repeat = expected_K // actual_K
        idx = torch.arange(expected_K, dtype=torch.long, device=block.data.device) % actual_K
        del repeat
        new_data = block.data[..., idx]
        block = Tensor(new_data, batch_ndim=block.batch_ndim, sub_batch_ndim=block.sub_batch_ndim)

    # Row sub_batch handling.
    have_row_sub = block.sub_batch_ndim > 0
    if row_structure == "dense" and have_row_sub:
        # Fold sub_row into the row_base axis.
        # Shape: (*dyn, *sub_row, row_base, K) -> (*dyn, sub_row_total*row_base, K)
        block = block.flatten_sub_batch_into_first_base_axis()
        have_row_sub = False

    # Col sub_batch handling: if col_structure == "block", unflatten K into
    # (*sub_col, col_base) so col intmd dims appear between row and col
    # base. The K trailing axis equals col_sb_total * col_var_base.
    if col_structure == "block" and col_sb:
        # block.data shape: (*dyn, [*sub_row,] row_storage, K=sub_col_total*col_base)
        # Reshape K -> (*sub_col, col_base) so the trailing 2+len(sub_col)
        # axes become (row_base, *sub_col, col_base). Then permute to swap
        # row_base behind sub_col, giving (*dyn, [*sub_row,] *sub_col,
        # row_base, col_base).
        old_shape = block.data.shape
        K_dim = old_shape[-1]
        if K_dim != col_sb_total * col_var_base:
            raise ValueError(
                f"K dim {K_dim} != sub_col_total({col_sb_total}) * col_base({col_var_base})"
            )
        col_intmd_shape = tuple(int(s) for s in col_sb)
        # After reshape, axes look like:
        #   (..., row_base, sub_col_0, sub_col_1, ..., col_base)
        # row_base sits at axis -(2 + len(col_intmd_shape)) of the new tensor.
        new_shape = (*old_shape[:-1], *col_intmd_shape, col_var_base)
        new_data = block.data.reshape(new_shape)
        ndim = new_data.ndim
        row_base_axis = ndim - 2 - len(col_intmd_shape)
        # Build permutation that moves row_base from row_base_axis to
        # position right BEFORE col_base (i.e. ndim - 2). Sub_col axes
        # shift left by one to fill the gap.
        perm = list(range(ndim))
        perm.pop(row_base_axis)
        perm.insert(ndim - 2, row_base_axis)
        new_data = new_data.permute(perm).contiguous()
        block = Tensor(
            new_data,
            batch_ndim=block.batch_ndim,
            sub_batch_ndim=block.sub_batch_ndim + len(col_intmd_shape),
        )

    # NB: we do NOT broadcast to row_batch_dyn here. The tangent's natural
    # batch_shape is what the chain rule produced; cross-block alignment is
    # handled by torch broadcasting at matmul / cat time. row_batch_dyn was
    # only ever a hint derived from one variable's `like` template; forcing
    # the tangent into it caused size-6 vs size-1 conflicts when the
    # tangent had legitimate dyn axes from upstream chain-rule combines.
    del row_batch_dyn
    return block


def _intmd_sum_k(r: Tensor, col_layout: AxisLayout, k: int) -> Tensor:
    """Reduce along the inner-group k's intmd axes on a matmul result.

    For BLOCK k_group, the post-mm tensor has the col-side intmd axes
    that were inherited from the k group's BLOCK structure. Sum them out.
    The intmd axes contributed by the k group are the FIRST k_sub_ndim
    axes of the result's sub_batch region (because the matmul preserved
    them through broadcasting / inheritance).
    """
    k_sb = col_layout.group_sub_batch_shape(k)
    n = len(k_sb)
    if n == 0 or r.sub_batch_ndim == 0:
        return r
    # Determine which sub_batch axes correspond to the k group.
    # Conservative: sum the last n sub_batch axes of r.
    new_data = r.data
    sb_start = new_data.ndim - r.data.ndim + r.batch_ndim
    del sb_start
    # The sub axes sit at positions [batch_ndim : batch_ndim + sub_batch_ndim].
    # Sum the LAST n of those (they came from the col k via broadcast).
    axes_to_sum = list(range(r.batch_ndim + r.sub_batch_ndim - n, r.batch_ndim + r.sub_batch_ndim))
    summed = new_data.sum(dim=axes_to_sum)
    return Tensor(
        summed,
        batch_ndim=r.batch_ndim,
        sub_batch_ndim=r.sub_batch_ndim - n,
    )


def _broadcast_batch(t: Tensor, target_dyn: tuple[int, ...]) -> Tensor:
    """Broadcast the batch region to ``target_dyn``.

    If ``t`` already has the right shape, return unchanged. Excess leading
    dims are squeezed when size-1; non-singleton extras raise.
    """
    cur = tuple(t.batch_shape)
    if cur == target_dyn:
        return t
    # Add leading size-1 axes if target has more dims.
    if len(cur) < len(target_dyn):
        pad = len(target_dyn) - len(cur)
        new_data = t.data
        for _ in range(pad):
            new_data = new_data.unsqueeze(0)
        return Tensor(
            new_data.expand(*target_dyn, *new_data.shape[len(target_dyn) :]).contiguous(),
            batch_ndim=len(target_dyn),
            sub_batch_ndim=t.sub_batch_ndim,
        )
    # Trim leading size-1 axes.
    n_extra = len(cur) - len(target_dyn)
    new_data = t.data
    for _ in range(n_extra):
        if new_data.shape[0] != 1:
            raise ValueError(
                f"_broadcast_batch: extra leading dim of size {new_data.shape[0]} "
                f"cannot be trimmed (cur={cur}, target={target_dyn})"
            )
        new_data = new_data.squeeze(0)
    new_data = new_data.expand(*target_dyn, *new_data.shape[len(target_dyn) :]).contiguous()
    return Tensor(
        new_data,
        batch_ndim=len(target_dyn),
        sub_batch_ndim=t.sub_batch_ndim,
    )


def _cat_along_base(parts: list[Tensor]) -> Tensor:
    """Concatenate Tensors along the LAST base axis. All inputs share
    batch + sub_batch + non-last-base shapes; only the trailing axis varies.
    """
    if not parts:
        raise ValueError("_cat_along_base: empty parts list")
    if len(parts) == 1:
        return parts[0]
    # Parts may have different ndims (zero blocks carry the row_batch_dyn
    # leading axes, non-zero tangent blocks inherit the chain-rule's
    # natural dyn shape). Left-pad each to the max ndim with size-1 axes,
    # then broadcast everything except the trailing cat axis.
    max_ndim = max(p.data.ndim for p in parts)
    padded_data = []
    for p in parts:
        d = p.data
        while d.ndim < max_ndim:
            d = d.unsqueeze(0)
        padded_data.append(d)
    try:
        target = torch.broadcast_shapes(*(d.shape[:-1] for d in padded_data))
    except RuntimeError as e:
        shapes = [tuple(p.data.shape) for p in parts]
        raise RuntimeError(
            f"_cat_along_base: broadcast failed with parts shapes {shapes}; original error: {e}"
        ) from e
    broadcasted = [d.expand(*target, d.shape[-1]).contiguous() for d in padded_data]
    cat_data = torch.cat(broadcasted, dim=-1)
    # batch_ndim grows to match max_ndim - sub_batch_ndim - base_ndim
    # (here base_ndim is the trailing axis we cat on; sub_batch_ndim is
    # whatever the parts share).
    parts_sub = parts[0].sub_batch_ndim
    # Result base_ndim: base axes are everything after dyn + sub. The
    # cat axis is the LAST base axis; preserve part's base_ndim count.
    parts_base = parts[0].base_ndim
    new_batch_ndim = cat_data.ndim - parts_sub - parts_base
    return Tensor(cat_data, batch_ndim=new_batch_ndim, sub_batch_ndim=parts_sub)


def _cat_along_row(parts: list[Tensor]) -> Tensor:
    """Concatenate Tensors along the row axis (-2 in base, base_ndim>=2).

    Left-pads parts of different ndim with size-1 leading axes (same
    ragged-ndim tolerance as :func:`_cat_along_base`).
    """
    if not parts:
        raise ValueError("_cat_along_row: empty parts list")
    if len(parts) == 1:
        return parts[0]
    max_ndim = max(p.data.ndim for p in parts)
    padded_data = []
    for p in parts:
        d = p.data
        while d.ndim < max_ndim:
            d = d.unsqueeze(0)
        padded_data.append(d)
    target = torch.broadcast_shapes(*(d.shape[:-2] + d.shape[-1:] for d in padded_data))
    bcs = []
    for d in padded_data:
        s = d.shape
        full = (*target[:-1], s[-2], target[-1])
        bcs.append(d.expand(full).contiguous())
    cat_data = torch.cat(bcs, dim=-2)
    parts_sub = parts[0].sub_batch_ndim
    parts_base = parts[0].base_ndim
    new_batch_ndim = cat_data.ndim - parts_sub - parts_base
    return Tensor(cat_data, batch_ndim=new_batch_ndim, sub_batch_ndim=parts_sub)


__all__ = [
    "AssembledVector",
    "AssembledMatrix",
    "norm",
    "norm_sq",
    "wrap_group_raw",
    "wrap_block_raw",
    "group_block_sub_batch_ndim",
    "_build_block_matrix",
]
