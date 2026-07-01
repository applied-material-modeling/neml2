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
from pyzag.chunktime import BidiagonalForwardOperator
from pyzag.operators.base import BlockJacobian, BlockVector

from neml2.es import AssembledMatrix
from neml2.types import Tensor
from neml2.types._boundary import to_torch

from ._assembly import _select_dynamic_am, _transpose_am
from ._flat import _layout_flat_size, _split_flat_to_av
from ._operator import NEML2SolvableBlockOperator
from ._vector import NEML2BlockVector


def _flip_time(am: AssembledMatrix) -> AssembledMatrix:
    """Flip every block along the leading dynamic (time) axis."""
    blocks = [
        [
            Tensor(
                to_torch(t).flip(0),
                batch_ndim=t.batch_ndim,
                sub_batch_ndim=t.sub_batch_ndim,
            )
            for t in row
        ]
        for row in am.tensors
    ]
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
        self.diag_am = diag_am
        self.sub_am = sub_am
        self._layout = layout
        self._reversed = _reversed

    def _first_diag_raw(self) -> torch.Tensor:
        return to_torch(self.diag_am.tensors[0][0])

    @property
    def device(self) -> torch.device:
        return self._first_diag_raw().device

    @property
    def dtype(self) -> torch.dtype:
        return self._first_diag_raw().dtype

    @property
    def nblk_steps(self) -> int:
        return self._first_diag_raw().shape[0]

    @property
    def batch_size(self) -> int:
        return self._first_diag_raw().shape[1]

    @property
    def block_size(self) -> int:
        return _layout_flat_size(self._layout)

    def _walk_diag(self) -> AssembledMatrix:
        return self.diag_am if not self._reversed else _flip_time(self.diag_am)

    def _walk_sub(self) -> AssembledMatrix:
        return self.sub_am if not self._reversed else _flip_time(self.sub_am)

    def forward_system(self, inverse_operator):
        if self._reversed:
            raise RuntimeError(
                "forward_system() must be called on a forward-walk BlockJacobian, "
                "not one returned by as_adjoint_walk()."
            )
        A_ops = NEML2SolvableBlockOperator.factored(self.diag_am)
        B_ops = NEML2SolvableBlockOperator(_select_dynamic_am(self.sub_am, slice(1, None)))
        return BidiagonalForwardOperator(A_ops, B_ops, inverse_operator=inverse_operator)

    def adjoint_system(self, inverse_operator):
        if not self._reversed:
            raise RuntimeError(
                "adjoint_system() must be called on the BlockJacobian returned by "
                "as_adjoint_walk(), not the forward one."
            )
        diag_walk = self._walk_diag()
        sub_walk = self._walk_sub()
        A_T = _transpose_am(_select_dynamic_am(diag_walk, slice(1, None)))
        B_T = _transpose_am(_select_dynamic_am(sub_walk, slice(1, -1)))
        A_ops = NEML2SolvableBlockOperator.factored(A_T)
        B_ops = NEML2SolvableBlockOperator(B_T)
        return inverse_operator(A_ops, B_ops)

    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> NEML2BlockVector:
        terminal = _select_dynamic_am(self.diag_am, slice(-1, None))
        op = NEML2SolvableBlockOperator.factored(_transpose_am(terminal))
        g_bv = NEML2BlockVector.from_av(_split_flat_to_av(g_terminal.unsqueeze(0), self._layout))
        sol = op.solve(g_bv)
        return NEML2BlockVector([-t for t in sol.raw_tensors], sol.layout, sol.intmd_dims)

    def couple_prev_chunk(self, a_first: BlockVector) -> NEML2BlockVector:
        if not isinstance(a_first, NEML2BlockVector):
            raise TypeError("NEML2BlockJacobian.couple_prev_chunk expects NEML2BlockVector.")
        sub_walk = self._walk_sub()
        boundary = _select_dynamic_am(sub_walk, slice(0, 1))
        op = NEML2SolvableBlockOperator(boundary)
        return op.t_matvec(a_first)

    def as_adjoint_walk(self) -> NEML2BlockJacobian:
        return NEML2BlockJacobian(
            self.diag_am, self.sub_am, self._layout, _reversed=not self._reversed
        )
