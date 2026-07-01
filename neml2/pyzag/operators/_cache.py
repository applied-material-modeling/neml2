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

"""Single-group LU solver that caches one factorization across repeated solves."""

from __future__ import annotations

import torch

from neml2.es import AssembledMatrix, AssembledVector
from neml2.types import Tensor
from neml2.types._boundary import to_torch


class CachingLU:
    """Single-group LU linear solver, drop-in for :class:`~neml2.solvers.DenseLU`.

    Factorizes the diagonal block once and reuses that factorization while the
    same block is solved against multiple right-hand sides. When injected as the
    ``primary_solver`` of a :class:`~neml2.solvers.SchurComplement`, this reuses
    the primary-block factorization across the two solves the Schur step makes
    against it (``A_pp^{-1} A_ps`` and ``A_pp^{-1} b_p``).
    """

    def __init__(self) -> None:
        self._key = None
        self._lu = None
        self._piv = None

    def _factor(self, block: Tensor):
        raw = to_torch(block)
        key = id(raw)
        if key != self._key:
            self._lu, self._piv = torch.linalg.lu_factor(raw)
            self._key = key
        return self._lu, self._piv

    def _lu_solve(self, lu, piv, rhs: Tensor) -> Tensor:
        raw = to_torch(rhs)
        is_vector = raw.ndim == lu.ndim - 1
        stacked = raw.unsqueeze(-1) if is_vector else raw
        sol = torch.linalg.lu_solve(lu, piv, stacked)
        if is_vector:
            sol = sol.squeeze(-1)
        return Tensor(sol, batch_ndim=rhs.batch_ndim, sub_batch_ndim=rhs.sub_batch_ndim)

    def solve(self, A: AssembledMatrix, b):
        """Solve ``A x = b`` for a single-group ``A`` (vector or matrix RHS)."""
        if A.row_layout.ngroup != 1 or A.col_layout.ngroup != 1:
            raise ValueError("CachingLU only supports single-group layouts")
        lu, piv = self._factor(A.tensors[0][0])
        if isinstance(b, AssembledVector):
            return AssembledVector(A.col_layout, [self._lu_solve(lu, piv, b.tensors[0])])
        return AssembledMatrix(
            A.col_layout, b.col_layout, [[self._lu_solve(lu, piv, b.tensors[0][0])]]
        )
