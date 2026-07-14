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

"""Python-native equation-system assembly for implicit updates.

Layered:

- :mod:`._helpers` -- raw-tensor helpers shared across the layer.
- :mod:`.axis_layout` -- :class:`AxisLayout`.
- :mod:`.assembled` -- :class:`AssembledVector`, :class:`AssembledMatrix`,
  :func:`norm` / :func:`norm_sq` (forwarded to :class:`~neml2.types.Tensor`).
- :mod:`.sparse` -- :class:`SparseVector`, :class:`SparseMatrix` (typed
  per-variable duals of the assembled forms; the symmetric API
  boundary surface).
- :mod:`.system` -- :class:`LinearSystem`, :class:`NonlinearSystem`,
  :class:`ModelNonlinearSystem`.
- :mod:`.implicit` -- AOTI export wrappers for the implicit-segment Newton
  path: operator graphs (:class:`RHS`, :class:`Jacobian`, :class:`JacobianGiven`,
  :class:`DrDParam`) + solve graphs (:class:`LinearSolve`, :class:`LinearSolveIFT`,
  :class:`LinearSolveParam`); the linear solve is un-baked from the operators.
"""

from .assembled import AssembledMatrix, AssembledVector, norm, norm_sq
from .axis_layout import AxisLayout
from .implicit import (
    RHS,
    DrDParam,
    Jacobian,
    JacobianGiven,
    LinearSolve,
    LinearSolveIFT,
    LinearSolveParam,
    Matvec,
)
from .sparse import SparseMatrix, SparseVector
from .system import LinearSystem, ModelNonlinearSystem, NonlinearSystem

__all__ = [
    "AxisLayout",
    "AssembledVector",
    "AssembledMatrix",
    "SparseVector",
    "SparseMatrix",
    "LinearSystem",
    "NonlinearSystem",
    "ModelNonlinearSystem",
    "RHS",
    "Jacobian",
    "Matvec",
    "LinearSolve",
    "JacobianGiven",
    "LinearSolveIFT",
    "DrDParam",
    "LinearSolveParam",
    "norm",
    "norm_sq",
]
