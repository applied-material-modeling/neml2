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
from neml2.solvers import LUCache, SchurComplement
from neml2.types import Tensor

from ._assembly import _require_le_one_intmd
from ._cache import CachingLU
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
        """Wrap the pyzag dense PCR state together with the neml2 row/col layouts."""
        self.dense_op = dense_op
        self.dense_state = dense_state
        self.row_layout = row_layout
        self.col_layout = col_layout


def _is_block(layout, g: int) -> bool:
    """True if group ``g`` of ``layout`` is a per-site (BLOCK) group."""
    return layout.structure[g] == "block"


class NEML2SolvableBlockOperator(SolvableBlockOperator):
    """Block operator backed by a neml2 ``AssembledMatrix``."""

    def __init__(self, am: AssembledMatrix) -> None:
        """Wrap an ``AssembledMatrix``; the LU factorization is cached lazily."""
        self.am = am
        self._lu = None
        self._piv = None
        # _lu / _piv are kept as working copies alongside the cache because
        # __getitem__ slices them along time for the Thomas sweep.
        self._lu_cache = LUCache(factor_fn=_lu_factor_guarded)

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
        raw = self.am.tensors[0][0].data  # data-ok pyzag boundary
        self._lu, self._piv = self._lu_cache.factor(raw)

    @property
    def device(self) -> torch.device:
        """Device of the backing tensors."""
        return self.am.tensors[0][0].device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the backing tensors."""
        return self.am.tensors[0][0].dtype

    @property
    def nblk(self) -> int:
        """Number of blocks along the dynamic (time) axis."""
        return self.am.tensors[0][0].batch_shape[0]

    @property
    def batch_size(self) -> int:
        """Size of the plain (non-dynamic) batch axis."""
        return self.am.tensors[0][0].batch_shape[1]

    def matvec(self, x: BlockVector) -> NEML2BlockVector:
        """Grain-diagonal matrix-vector product ``self @ x``.

        Delegates to :meth:`~neml2.es.AssembledMatrix.per_instance_matvec` (a
        BLOCK output stays per-site; only a DENSE output consuming a BLOCK input
        aggregates the site axis). Raw tensors appear only at the final
        :class:`NEML2BlockVector` hand-off.
        """
        if not isinstance(x, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.matvec expects NEML2BlockVector.")
        _require_le_one_intmd(self.am, "matvec")
        return NEML2BlockVector.from_av(self.am.per_instance_matvec(x.to_av()))

    def t_matvec(self, x: BlockVector) -> NEML2BlockVector:
        """Transposed grain-diagonal matrix-vector product ``self.T @ x`` (see :meth:`matvec`)."""
        if not isinstance(x, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.t_matvec expects NEML2BlockVector.")
        _require_le_one_intmd(self.am, "matvec")
        return NEML2BlockVector.from_av(self.am.per_instance_matvec(x.to_av(), transpose=True))

    def _primary_group(self) -> int:
        """Index of the per-site (BLOCK) group to use as the Schur primary; 0 if none."""
        for g in range(self.am.row_layout.ngroup):
            if _is_block(self.am.row_layout, g):
                return g
        return 0

    def solve(self, rhs: BlockVector) -> NEML2BlockVector:
        """Solve ``self @ x = rhs`` for the diagonal block.

        Single-group layouts use the cached batched LU; a two-group BLOCK+DENSE
        split delegates to :class:`~neml2.solvers.SchurComplement` with a
        :class:`~neml2.pyzag.operators.CachingLU` primary solver. Three or more
        groups are unsupported.
        """
        if not isinstance(rhs, NEML2BlockVector):
            raise TypeError("NEML2SolvableBlockOperator.solve expects NEML2BlockVector.")
        ng = self.am.row_layout.ngroup
        if ng == 1:
            self._ensure_lu()
            assert self._lu is not None and self._piv is not None  # set by _ensure_lu for ngroup==1
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
        """Deep copy the operator, cloning every backing tensor (cache dropped)."""
        blocks = [[t.clone() for t in row] for row in self.am.tensors]
        return NEML2SolvableBlockOperator(
            AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        )

    def __getitem__(self, idx: int | slice) -> NEML2SolvableBlockOperator:
        """Slice along the dynamic (time) axis, carrying the cached factorization.

        An ``int`` index is normalized to a length-1 slice so the block axis is
        preserved and ``self.am`` stays in sync with the cached ``_lu`` / ``_piv``.
        """
        # Normalize an int index to a length-1 slice so the block (time) axis is
        # preserved on both the assembled matrix and the cached factorization; a
        # bare int would collapse it, desyncing self.am from _lu / _piv.
        if isinstance(idx, int):
            idx = slice(idx, idx + 1) if idx != -1 else slice(idx, None)
        sliced = NEML2SolvableBlockOperator(self.am.batch[idx])
        if self._lu is not None and self._piv is not None:
            sliced._lu = self._lu[idx]
            sliced._piv = self._piv[idx]
        return sliced

    def __setitem__(self, idx: int | slice, other: BlockOperator) -> None:
        """Write ``other`` into the ``idx`` slice of the dynamic axis; invalidates the cache."""
        if not isinstance(other, NEML2SolvableBlockOperator):
            raise TypeError(
                "NEML2SolvableBlockOperator assignment requires NEML2SolvableBlockOperator."
            )
        blocks = [
            [
                t_self.batch.set(idx, t_other)
                for t_self, t_other in zip(row_self, row_other, strict=True)
            ]
            for row_self, row_other in zip(self.am.tensors, other.am.tensors, strict=True)
        ]
        self.am = AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        self._lu = None
        self._piv = None
        self._lu_cache.invalidate()

    def pad_front(self, n: int = 1) -> NEML2SolvableBlockOperator:
        """Return a copy with ``n`` zero blocks prepended along the dynamic axis.

        Used to align a subdiagonal operator with the diagonal so block ``k`` of
        the padded result is the coupling into step ``k``.
        """
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        blocks = [[t.batch.pad(0, before=n) for t in row] for row in self.am.tensors]
        return NEML2SolvableBlockOperator(
            AssembledMatrix(self.am.row_layout, self.am.col_layout, blocks)
        )

    def _is_single_dense(self) -> bool:
        """True for a single-group DENSE layout (the only native PCR path here)."""
        return self.am.row_layout.ngroup == 1 and not _is_block(self.am.row_layout, 0)

    def pcr_init(self, B: BlockOperator, v: BlockVector) -> PCRState:
        """Seed a parallel-cyclic-reduction sweep (single-group dense only).

        Delegates to pyzag's dense backend. Raises ``NotImplementedError`` for
        multi-group / BLOCK layouts -- those are handled instead by
        :class:`~neml2.pyzag.operators.NEML2LiftedPCRFactorization`, dispatched
        through :class:`~neml2.pyzag.operators.NEML2BlockJacobian`.
        """
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
        dense_op = DenseBlockOperator(self.am.tensors[0][0].data)  # data-ok pyzag boundary
        dense_B = DenseBlockOperator(B.am.tensors[0][0].data)  # data-ok pyzag boundary
        dense_v = DenseBlockVector(v.raw_tensors[0])
        dense_state = dense_op.pcr_init(dense_B, dense_v)
        return _SingleGroupPCRState(dense_op, dense_state, self.am.row_layout, self.am.col_layout)

    def pcr_reduce_level(self, state: PCRState, level: int):
        """Advance the PCR sweep by one stride-doubling level (single-group dense)."""
        if not isinstance(state, _SingleGroupPCRState):
            raise TypeError("state must be a _SingleGroupPCRState from pcr_init.")
        new_state = state.dense_op.pcr_reduce_level(state.dense_state, level)
        return _SingleGroupPCRState(state.dense_op, new_state, state.row_layout, state.col_layout)

    def pcr_finalize(self, state: PCRState):
        """Finish the PCR sweep, returning the reduced ``(operator, rhs)`` pair."""
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
