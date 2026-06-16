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

"""Dense LU linear solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from neml2.es import AssembledMatrix, AssembledVector
from neml2.factory import register_neml2_object
from neml2.schema import HitSchema

if TYPE_CHECKING:
    import nmhit

    from neml2.factory import _NativeInputFile


@register_neml2_object("DenseLU")
class DenseLU:
    """Dense LU linear solver. Assembles the (possibly) sparse matrix into a
    dense one and uses a standard LU decomposition.

    The dense-vs-block-diagonal dispatch is implicit in the wrapped
    :class:`~neml2.types.Tensor` operand: ``Tensor.solve`` forwards to
    ``torch.linalg.solve`` on the trailing matrix dims, so leading
    sub-batch axes (if any) batch naturally as independent LUs per site
    -- no special-case code at this layer.
    """

    SECTION = "Solvers"

    # No tunable options; the empty schema documents that in the syntax catalog.
    hit = HitSchema()

    @classmethod
    def from_hit(cls, node: nmhit.Node, factory: _NativeInputFile) -> DenseLU:
        del node, factory
        return cls()

    @overload
    def solve(self, A: AssembledMatrix, b: AssembledVector) -> AssembledVector: ...
    @overload
    def solve(self, A: AssembledMatrix, b: AssembledMatrix) -> AssembledMatrix: ...
    def solve(self, A, b):
        if A.row_layout.ngroup != 1 or A.col_layout.ngroup != 1:
            raise ValueError("DenseLU only supports single-group layouts")
        A00 = A.tensors[0][0]
        if isinstance(b, AssembledVector):
            if b.layout.ngroup != 1:
                raise ValueError("DenseLU only supports single-group vector RHS")
            return AssembledVector(A.col_layout, [A00.solve(b.tensors[0])])
        if b.row_layout.ngroup != 1 or b.col_layout.ngroup != 1:
            raise ValueError("DenseLU only supports single-group matrix RHS")
        return AssembledMatrix(A.col_layout, b.col_layout, [[A00.solve(b.tensors[0][0])]])


__all__ = ["DenseLU"]
