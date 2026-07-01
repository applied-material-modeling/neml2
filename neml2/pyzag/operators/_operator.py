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

"""pyzag ``SolvableBlockOperator`` backed by a neml2 ``AssembledMatrix``.

The diagonal-block solve uses a cached batched LU factorization for a single
group -- factored once (:meth:`NEML2SolvableBlockOperator.factored`) and sliced
per block through a Thomas sweep -- and delegates a 2-group BLOCK+DENSE split to
:class:`~neml2.solvers.SchurComplement`. Matrix-vector products use
a per-instance interpretation (per-site diagonal for BLOCK groups, aggregation
only when a DENSE output consumes a BLOCK input), which is the mutual inverse of
the Schur solve and differs from neml2's native ``AssembledMatrix @`` (that
aggregates on every BLOCK contraction, correct only for the Schur cross-terms).
Parallel cyclic reduction is implemented for single-group dense layouts (via the
pyzag dense backend); other layouts use the Thomas factorization.
"""

from __future__ import annotations

from math import prod

import torch
from pyzag.operators.base import (
    BlockOperator,
    BlockVector,
    PCRState,
    SolvableBlockOperator,
)

# this is for PCR implementation without int md axis
from pyzag.operators.dense import (
    DenseBlockOperator,
    DenseBlockVector,
    _lu_factor_guarded,
    batch_lu_solve,
)

from neml2.es import AssembledMatrix
from neml2.solvers import SchurComplement
from neml2.types import Tensor
from neml2.types._boundary import to_torch

from ._assembly import _require_le_one_intmd, _select_dynamic_am
from ._cache import CachingLU
from ._flat import _group_flat_size
from ._vector import NEML2BlockVector

_PCR_MESSAGE = (
    "PCR for the neml2 backend is only implemented for single-group dense layouts; "
    "multi-group / BLOCK (structured Schur-PCR) is not supported. Use the Thomas "
    "factorization (chunktime.BidiagonalThomasFactorization, the solver default)."
)


# Single group means no int md axis
class _SingleGroupPCRState(PCRState):
    """Opaque PCR state for the single-group dense path."""

    def __init__(self, dense_op, dense_state, row_layout, col_layout) -> None:
        self.dense_op = dense_op
        self.dense_state = dense_state
        self.row_layout = row_layout
        self.col_layout = col_layout


def _is_block(layout, g: int) -> bool:
    return layout.structure[g] == "block"


def _group_intmd_sizes(layout, g: int) -> list[int]:
    if not _is_block(layout, g):
        return []
    return [int(s) for s in layout.group_sub_batch_shape(g)]


class NEML2SolvableBlockOperator(SolvableBlockOperator):
    """Block operator backed by a neml2 ``AssembledMatrix``."""

    def __init__(self, am: AssembledMatrix) -> None:
        self.am = am
        self._lu = None
        self._piv = None

    @classmethod
    def factored(cls, am: AssembledMatrix) -> NEML2SolvableBlockOperator:
        """Construct the operator and eagerly cache its factorization.

        For a single-group diagonal this factors all block-axis entries at once
        (one batched ``lu_factor``); the cache is carried through
        :meth:`__getitem__` so a Thomas sweep reuses it per block via
        ``lu_solve`` instead of re-factoring. Multi-group (Schur) diagonals are
        not cached here -- they delegate to :class:`SchurComplement` per solve.
        """
        op = cls(am)
        op._ensure_lu()
        return op

    def _ensure_lu(self) -> None:
        """Cache the batched LU of a single-group diagonal block (no-op otherwise).

        Uses pyzag's ``_lu_factor_guarded`` so large blocks on CUDA fall back to
        a per-matrix loop rather than cuSOLVER's slow batched path.
        """
        if self._lu is not None:
            return
        if self.am.row_layout.ngroup != 1:
            return
        self._lu, self._piv = _lu_factor_guarded(to_torch(self.am.tensors[0][0]))

    @property
    def device(self) -> torch.device:
        return to_torch(self.am.tensors[0][0]).device

    @property
    def dtype(self) -> torch.dtype:
        return to_torch(self.am.tensors[0][0]).dtype

    @property
    def nblk(self) -> int:
        return to_torch(self.am.tensors[0][0]).shape[0]

    @property
    def batch_size(self) -> int:
        return to_torch(self.am.tensors[0][0]).shape[1]

    def matvec(self, x: BlockVector) -> NEML2BlockVector:
        if not isinstance(x, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.matvec expects NEML2BlockVector.")
        return self._mv_per_instance(self.am, x, transpose=False)

    def t_matvec(self, x: BlockVector) -> NEML2BlockVector:
        if not isinstance(x, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.t_matvec expects NEML2BlockVector.")
        return self._mv_per_instance(self.am, x, transpose=True)

    @staticmethod
    def _mv_per_instance(am, x, transpose: bool):
        """Per-instance matrix-vector product (matvec or its transpose).

        Deliberately not neml2's ``AssembledMatrix @``: that unconditionally sums
        over a contracted BLOCK group's site axis (correct only for the Schur
        cross-terms). Here a BLOCK output stays per-site -- the grain-diagonal
        ``B @ v`` that Thomas and adjoint coupling need. The site sum fires only
        when a DENSE output consumes a BLOCK input -- in Schur terms the
        ``A_sp @ x_p`` term (DENSE row x BLOCK col) that aggregates the per-grain
        contributions into the global (Schur) residual ``b_s``.

        The two loops range over variable *groups* (typically 1, or 2 for a
        BLOCK+DENSE split) -- not over sites or time steps.
        """
        _require_le_one_intmd(am, "matvec")
        if transpose:
            out_layout = am.col_layout
            in_layout = am.row_layout
        else:
            out_layout = am.row_layout
            in_layout = am.col_layout

        n_out = out_layout.ngroup
        n_in = in_layout.ngroup
        out_tensors: list[torch.Tensor | None] = [None] * n_out

        for i in range(n_out):
            out_is_block = _is_block(out_layout, i)
            accum: torch.Tensor | None = None

            for j in range(n_in):
                blk = am.tensors[j][i] if transpose else am.tensors[i][j]
                blk_raw = to_torch(blk)
                if transpose:
                    blk_raw = blk_raw.transpose(-1, -2)

                xj = x.raw_tensors[j]
                x_intmd = x.intmd_dims[j]
                blk_intmd = blk.sub_batch_ndim

                diff = blk_intmd - x_intmd
                if diff > 0:
                    for _ in range(diff):
                        xj = xj.unsqueeze(2)
                elif diff < 0:
                    for _ in range(-diff):
                        blk_raw = blk_raw.unsqueeze(2)
                effective_intmd = max(blk_intmd, x_intmd)

                r = torch.matmul(blk_raw, xj.unsqueeze(-1)).squeeze(-1)

                if not out_is_block and effective_intmd > 0:
                    for _ in range(effective_intmd):
                        r = r.sum(dim=-2)

                if accum is None:
                    accum = r
                else:
                    if r.ndim != accum.ndim:
                        if r.ndim < accum.ndim:
                            for _ in range(accum.ndim - r.ndim):
                                r = r.unsqueeze(2)
                        else:
                            for _ in range(r.ndim - accum.ndim):
                                accum = accum.unsqueeze(2)
                    accum = accum + r

            if accum is None:
                intmd_sizes = _group_intmd_sizes(out_layout, i) if out_is_block else []
                ref_shape = to_torch(am.tensors[0][0]).shape
                nblk, sbat = ref_shape[0], ref_shape[1]
                group_dofs = _group_flat_size(out_layout, i)
                intmd_numel = int(prod(intmd_sizes)) if intmd_sizes else 1
                n_out_var = group_dofs // intmd_numel
                accum = torch.zeros(
                    (nblk, sbat, *intmd_sizes, n_out_var),
                    dtype=x.raw_tensors[0].dtype,
                    device=x.raw_tensors[0].device,
                )
            out_tensors[i] = accum

        out_intmd_dims = [max(0, t.ndim - 3) for t in out_tensors]
        return NEML2BlockVector(out_tensors, out_layout, out_intmd_dims)

    def _primary_group(self) -> int:
        for g in range(self.am.row_layout.ngroup):
            if _is_block(self.am.row_layout, g):
                return g
        return 0

    def solve(self, rhs: BlockVector) -> NEML2BlockVector:
        if not isinstance(rhs, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.solve expects NEML2BlockVector.")
        ng = self.am.row_layout.ngroup
        if ng == 1:
            self._ensure_lu()
            raw = rhs.raw_tensors[0]
            x = batch_lu_solve(self._lu, self._piv, raw.unsqueeze(-1)).squeeze(-1)
            return NEML2BlockVector([x], self.am.col_layout, list(rhs.intmd_dims))
        if ng == 2:
            primary = self._primary_group()
            solver = SchurComplement(
                residual_primary_group=primary,
                unknown_primary_group=primary,
                primary_solver=CachingLU(),
            )
            out = solver.solve(self.am, rhs.to_av())
        else:
            raise NotImplementedError(f"neml2 backend solve supports 1 or 2 groups (got {ng}).")
        return NEML2BlockVector.from_av(out)

    def clone(self) -> NEML2SolvableBlockOperator:
        blocks = [
            [
                Tensor(
                    to_torch(t).clone(),
                    batch_ndim=t.batch_ndim,
                    sub_batch_ndim=t.sub_batch_ndim,
                )
                for t in row
            ]
            for row in self.am.tensors
        ]
        return NEML2SolvableBlockOperator(
            AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        )

    def __getitem__(self, idx: int | slice) -> NEML2SolvableBlockOperator:
        # Normalize an int index to a length-1 slice so the block (time) axis is
        # preserved on both the assembled matrix and the cached factorization; a
        # bare int would collapse it, desyncing self.am from _lu / _piv.
        if isinstance(idx, int):
            idx = slice(idx, idx + 1) if idx != -1 else slice(idx, None)
        sliced = NEML2SolvableBlockOperator(_select_dynamic_am(self.am, idx))
        if self._lu is not None:
            sliced._lu = self._lu[idx]
            sliced._piv = self._piv[idx]
        return sliced

    def __setitem__(self, idx: int | slice, other: BlockOperator) -> None:
        if not isinstance(other, NEML2SolvableBlockOperator):
            raise TypeError(
                "NEML2SolvableBlockOperator assignment requires NEML2SolvableBlockOperator."
            )
        blocks = []
        for row_self, row_other in zip(self.am.tensors, other.am.tensors, strict=True):
            new_row = []
            for t_self, t_other in zip(row_self, row_other, strict=True):
                new_raw = to_torch(t_self).clone()
                new_raw[idx] = to_torch(t_other)
                new_row.append(
                    Tensor(
                        new_raw,
                        batch_ndim=t_self.batch_ndim,
                        sub_batch_ndim=t_self.sub_batch_ndim,
                    )
                )
            blocks.append(new_row)
        self.am = AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        self._lu = None
        self._piv = None

    def pad_front(self, n: int = 1) -> NEML2SolvableBlockOperator:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        blocks = []
        for row in self.am.tensors:
            new_row = []
            for t in row:
                raw = to_torch(t)
                raw_pad = torch.nn.functional.pad(raw, (0,) * (2 * (raw.ndim - 1)) + (n, 0))
                new_row.append(
                    Tensor(
                        raw_pad,
                        batch_ndim=t.batch_ndim,
                        sub_batch_ndim=t.sub_batch_ndim,
                    )
                )
            blocks.append(new_row)
        return NEML2SolvableBlockOperator(
            AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        )

    def trim_front(self, n: int = 1) -> NEML2SolvableBlockOperator:
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        return self[n:]

    def _is_single_dense(self) -> bool:
        return self.am.row_layout.ngroup == 1 and not _is_block(self.am.row_layout, 0)

    def pcr_init(self, B: BlockOperator, v: BlockVector) -> PCRState:
        if not self._is_single_dense():
            raise NotImplementedError(_PCR_MESSAGE)
        if not isinstance(B, NEML2SolvableBlockOperator):
            raise TypeError("PCR sub-operator B must be a NEML2SolvableBlockOperator.")
        if not isinstance(v, NEML2BlockVector):
            raise TypeError("PCR rhs v must be a NEML2BlockVector.")
        if (
            B.am.row_layout.ngroup != 1
            or B.am.col_layout.ngroup != 1
            or v.layout.ngroup != 1
            or _is_block(B.am.row_layout, 0)
            or _is_block(B.am.col_layout, 0)
            or v.intmd_dims[0] != 0
        ):
            raise NotImplementedError(_PCR_MESSAGE)
        dense_op = DenseBlockOperator(to_torch(self.am.tensors[0][0]))
        dense_B = DenseBlockOperator(to_torch(B.am.tensors[0][0]))
        dense_v = DenseBlockVector(v.raw_tensors[0])
        dense_state = dense_op.pcr_init(dense_B, dense_v)
        return _SingleGroupPCRState(dense_op, dense_state, self.am.row_layout, self.am.col_layout)

    def pcr_reduce_level(self, state: PCRState, level: int):
        if not isinstance(state, _SingleGroupPCRState):
            raise TypeError("state must be a _SingleGroupPCRState from pcr_init.")
        new_state = state.dense_op.pcr_reduce_level(state.dense_state, level)
        return _SingleGroupPCRState(state.dense_op, new_state, state.row_layout, state.col_layout)

    def pcr_finalize(self, state: PCRState):
        if not isinstance(state, _SingleGroupPCRState):
            raise TypeError("state must be a _SingleGroupPCRState from pcr_init.")
        B_red, v_red = state.dense_op.pcr_finalize(state.dense_state)
        B_raw = B_red.data
        B_red_am = AssembledMatrix(
            state.row_layout,
            state.col_layout,
            [[Tensor(B_raw, batch_ndim=B_raw.ndim - 2, sub_batch_ndim=0)]],
        )
        v_red_bv = NEML2BlockVector([v_red.data], state.col_layout, [0])
        return NEML2SolvableBlockOperator(B_red_am), v_red_bv
