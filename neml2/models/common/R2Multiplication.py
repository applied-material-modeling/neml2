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

"""Python-native mirror of C++ ``common/R2Multiplication.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import R2, inv


@register_native("R2Multiplication")
class R2Multiplication(Model):
    r"""Multiplication of form ``A B``, where $A$ and $B$ are second order
    tensors. A and B can be inverted and/or transposed per request.

    Mirrors the C++ ``R2Multiplication`` model
    (``include/neml2/models/common/R2Multiplication.h``). The forward composes
    the (optionally inverted and/or transposed) operands via the R2 ``@``
    matmul. The pushforward applies the product rule to the matmul and,
    when an operand is inverted, the matrix-inverse differential identity
    $d(X^-1) = -X^-1 dX X^-1$ -- all expressed in typed R2 algebra, no
    Jacobian materialised.
    """

    hit = HitSchema(
        input("A", R2, "Variable A"),
        input("B", R2, "Variable B"),
        output("to", R2, "The result of the multiplication"),
        option(
            "invert_A",
            bool,
            "Whether to invert A",
            default=False,
            attr="_invA",
        ),
        option(
            "invert_B",
            bool,
            "Whether to invert B",
            default=False,
            attr="_invB",
        ),
        option(
            "transpose_A",
            bool,
            "Whether to transpose A",
            default=False,
            attr="_transA",
        ),
        option(
            "transpose_B",
            bool,
            "Whether to transpose B",
            default=False,
            attr="_transB",
        ),
    )

    # The four bool options land on these attributes via ``attr=``. Annotate so
    # pyright sees plain ``bool``s instead of nn.Module's ``Module`` hint that
    # ``Model.__getattr__`` would otherwise mask.
    _invA: bool
    _invB: bool
    _transA: bool
    _transB: bool

    def forward(  # type: ignore[override]
        self,
        A: R2,
        B: R2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # Apply the optional inverse / transpose to each operand. Composing
        # inverse-then-transpose is equivalent to transpose-then-inverse for a
        # nonsingular 3x3, so the order chosen here just matches the C++ code.
        Aop = inv(A) if self._invA else A
        if self._transA:
            Aop = Aop.base.transpose(-2, -1)

        Bop = inv(B) if self._invB else B
        if self._transB:
            Bop = Bop.base.transpose(-2, -1)

        C = Aop @ Bop
        if v is None:
            return C

        # Differential pushforward for each input. The differential of
        # ``op_X(X)`` applied to a tangent V is:
        #   - identity:          dop_X(V) = V
        #   - transpose only:    dop_X(V) = V.T
        #   - inverse only:      dop_X(V) = -X^-1 V X^-1 = -Xop V Xop
        #   - inverse+transpose: dop_X(V) = -Xop V.T Xop  (matrix-inverse
        #     differential composed with transpose: see e.g. Magnus & Neudecker).
        # Then ``dC = dop_A(V_A) @ Bop + Aop @ dop_B(V_B)`` by the product rule
        # on the outer matmul. Each input contributes independently.

        def A_action(V: R2) -> R2:
            dAop = V.base.transpose(-2, -1) if self._transA else V
            if self._invA:
                dAop = -(Aop @ dAop @ Aop)
            return dAop @ Bop

        def B_action(V: R2) -> R2:
            dBop = V.base.transpose(-2, -1) if self._transB else V
            if self._invB:
                dBop = -(Bop @ dBop @ Bop)
            return Aop @ dBop

        return C, self.apply_chain_rule(
            v,
            "to",
            {"A": A_action, "B": B_action},
            output=C,
        )


__all__ = ["R2Multiplication"]
