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

"""Python-native mirror of C++ ``common/SR2ToR2.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import (
    R2,
    SR2,
    r2_from_sr2,
)


@register_native("SR2ToR2")
class SR2ToR2(Model):
    """Convert a symmetric rank two tensor to a full tensor."""

    # Forward: ``y = R2(A)`` -- unpack the Mandel-packed SR2 into a full 3x3
    # R2 (off-diagonals divided by sqrt(2)). Linear leaf: the
    # directional pushforward along an SR2 tangent V is just ``r2_from_sr2(V)``
    # -- the same typed wrapper map applied to the tangent. No Jacobian
    # materialised; ``r2_from_sr2`` already lives in
    # ``neml2.types.functions`` so the body is pure wrapper algebra
    # with no ``torch.<op>(.data)`` calls.

    hit = HitSchema(
        input("input", SR2, "Symmetric second order tensor to convert"),
        output("output", R2, "Output full second order tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        A: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        y = r2_from_sr2(A)
        if v is None:
            return y

        def A_action(V: SR2) -> R2:
            return r2_from_sr2(V)

        return y, self.apply_chain_rule(v, "output", {"input": A_action}, output=y)


__all__ = ["SR2ToR2"]
