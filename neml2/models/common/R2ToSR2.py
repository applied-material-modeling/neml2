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

"""Python-native mirror of C++ ``common/R2ToSR2.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import (
    R2,
    SR2,
    sym,
)


@register_native("R2ToSR2")
class R2ToSR2(Model):
    """Extract the symmetric part of a R2 tensor."""

    # Forward: ``y = sym(A) = (A + A^T) / 2`` packed in Mandel form.
    # Linear leaf: the directional pushforward along an R2 tangent V
    # is just ``sym(V)`` -- the same typed wrapper map applied to the tangent.
    # No Jacobian materialised; ``sym`` already lives in
    # ``neml2.types.functions`` so the body is pure wrapper algebra
    # with no ``torch.<op>(.data)`` calls.

    hit = HitSchema(
        input("input", R2, "Second order tensor to split"),
        output("output", SR2, "Output symmetric second order tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        A: R2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        y = sym(A)
        if v is None:
            return y

        def A_action(V: R2) -> SR2:
            return sym(V)

        return y, self.apply_chain_rule(v, "output", {"input": A_action}, output=y)


__all__ = ["R2ToSR2"]
