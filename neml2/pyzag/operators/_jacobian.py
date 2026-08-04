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

"""pyzag bidiagonal ``BlockJacobian`` backed by neml2 ``AssembledMatrix`` blocks."""

from __future__ import annotations

import torch
from pyzag.chunktime import BidiagonalForwardOperator, BidiagonalPCRFactorization
from pyzag.operators.base import BlockJacobian, BlockVector

from neml2.es import AssembledMatrix, AssembledVector

from ._lifted_pcr import NEML2LiftedPCRFactorization
from ._operator import NEML2SolvableBlockOperator
from ._vector import NEML2BlockVector


def _has_block_group(layout) -> bool:
    """True if the layout has an intermediate (BLOCK / per-site) dimension."""
    return any(layout.structure[g] == "block" for g in range(layout.ngroup))


def _select_inverse(inverse_operator, layout):
    """Route PCR/Hybrid on a BLOCK layout to the lifted-arrowhead factorization.

    The dense per-site operator is never manifested for a layout with an
    intermediate dimension; single-group dense layouts keep pyzag's dense path.
    ``BidiagonalHybridFactorizationImpl`` is a subclass of
    ``BidiagonalPCRFactorization`` so both are covered when passed as a class.
    """
    if (
        isinstance(inverse_operator, type)
        and issubclass(inverse_operator, BidiagonalPCRFactorization)
        and _has_block_group(layout)
    ):
        return NEML2LiftedPCRFactorization
    return inverse_operator


def _flip_time(am: AssembledMatrix) -> AssembledMatrix:
    """Flip every block along the leading dynamic (time) axis."""
    blocks = [[t.batch.flip(0) for t in row] for row in am.tensors]
    return AssembledMatrix(am.row_layout, am.col_layout, blocks)


class NEML2BlockJacobian(BlockJacobian):
    """Per-chunk bidiagonal Jacobian wrapping neml2 ``AssembledMatrix`` diag/sub."""

    def __init__(
        self,
        diag_am: AssembledMatrix,
        sub_am: AssembledMatrix,
        layout,
        _reversed: bool = False,
    ) -> None:
        """Hold the diagonal (``dr/du``) and subdiagonal (``dr/du_old``) chunk blocks.

        Args:
            diag_am: the per-step diagonal ``AssembledMatrix``.
            sub_am: the per-step subdiagonal ``AssembledMatrix`` coupling to old state.
            layout: the state :class:`~neml2.es.axis_layout.AxisLayout`.
            _reversed: internal flag marking an adjoint (time-reversed) walk; set via
                :meth:`as_adjoint_walk`, not directly.
        """
        self.diag_am = diag_am
        self.sub_am = sub_am
        self._layout = layout
        self._reversed = _reversed

    @property
    def device(self) -> torch.device:
        """Device of the backing tensors."""
        return self.diag_am.tensors[0][0].device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the backing tensors."""
        return self.diag_am.tensors[0][0].dtype

    @property
    def nblk_steps(self) -> int:
        """Number of time steps (blocks) in this chunk."""
        return self.diag_am.tensors[0][0].batch_shape[0]

    @property
    def batch_size(self) -> int:
        """Size of the plain (non-dynamic) batch axis."""
        return self.diag_am.tensors[0][0].batch_shape[1]

    @property
    def block_size(self) -> int:
        """Total per-step degrees of freedom of the state layout."""
        return self._layout.flat_size()

    def _walk_diag(self) -> AssembledMatrix:
        """Diagonal blocks in walk order (time-flipped when reversed)."""
        return self.diag_am if not self._reversed else _flip_time(self.diag_am)

    def _walk_sub(self) -> AssembledMatrix:
        """Subdiagonal blocks in walk order (time-flipped when reversed)."""
        return self.sub_am if not self._reversed else _flip_time(self.sub_am)

    def forward_system(self, inverse_operator):
        """Build the forward bidiagonal chunk operator solved during the state sweep.

        ``inverse_operator`` is the factorization pyzag requests; a PCR/Hybrid
        request on a BLOCK layout is redirected to
        :class:`NEML2LiftedPCRFactorization` (see :func:`_select_inverse`). Must be
        called on a forward-walk Jacobian, not one from :meth:`as_adjoint_walk`.
        """
        if self._reversed:
            raise RuntimeError(
                "forward_system() must be called on a forward-walk BlockJacobian, "
                "not one returned by as_adjoint_walk()."
            )
        A_ops = NEML2SolvableBlockOperator.factored(self.diag_am)
        B_ops = NEML2SolvableBlockOperator(self.sub_am.batch[1:])
        inverse_operator = _select_inverse(inverse_operator, self._layout)
        return BidiagonalForwardOperator(
            A_ops,
            B_ops,
            inverse_operator=inverse_operator,  # type: ignore[arg-type]
        )

    # pyzag's abstract adjoint_system is unannotated (inferred -> None), so our
    # concrete operator return trips the override check.
    def adjoint_system(self, inverse_operator):  # type: ignore[override]
        """Build the transposed, time-reversed bidiagonal operator for the adjoint sweep.

        Uses the transposed diagonal/subdiagonal blocks in reverse time order. Must
        be called on the Jacobian returned by :meth:`as_adjoint_walk`. As with
        :meth:`forward_system`, PCR/Hybrid on a BLOCK layout routes to the lifted
        factorization.
        """
        if not self._reversed:
            raise RuntimeError(
                "adjoint_system() must be called on the BlockJacobian returned by "
                "as_adjoint_walk(), not the forward one."
            )
        diag_walk = self._walk_diag()
        sub_walk = self._walk_sub()
        A_T = diag_walk.batch[1:].transpose()
        B_T = sub_walk.batch[1:-1].transpose()
        A_ops = NEML2SolvableBlockOperator.factored(A_T)
        B_ops = NEML2SolvableBlockOperator(B_T)
        inverse_operator = _select_inverse(inverse_operator, self._layout)
        return inverse_operator(A_ops, B_ops)

    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> NEML2BlockVector:
        """Seed the adjoint recursion from the terminal cost gradient ``g_terminal``.

        Solves ``-A_last^T x = g_terminal`` on the last step's transposed diagonal.
        """
        terminal = self.diag_am.batch[-1:]
        op = NEML2SolvableBlockOperator.factored(terminal.transpose())
        g_av = AssembledVector.from_flat(self._layout, g_terminal.unsqueeze(0))
        g_bv = NEML2BlockVector.from_av(g_av)
        sol = op.solve(g_bv)
        return NEML2BlockVector([-t for t in sol.raw_tensors], sol.layout, sol.intmd_dims)

    def couple_prev_chunk(self, a_first: BlockVector) -> NEML2BlockVector:
        """Propagate the adjoint across the chunk boundary via the subdiagonal coupling.

        Applies the transpose of this chunk's boundary subdiagonal block to the
        first adjoint variable ``a_first``, giving the contribution into the
        previous chunk's terminal adjoint.
        """
        if not isinstance(a_first, NEML2BlockVector):
            raise TypeError("NEML2BlockJacobian.couple_prev_chunk expects NEML2BlockVector.")
        sub_walk = self._walk_sub()
        boundary = sub_walk.batch[0:1]
        op = NEML2SolvableBlockOperator(boundary)
        return op.t_matvec(a_first)

    def as_adjoint_walk(self) -> NEML2BlockJacobian:
        """Return a view of this Jacobian marked for the reverse-time adjoint walk."""
        return NEML2BlockJacobian(
            self.diag_am, self.sub_am, self._layout, _reversed=not self._reversed
        )
