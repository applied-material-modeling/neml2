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

"""Python-native mirror of C++ ``common/ScalarToDiagonalSR2.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import (
    SR2,
    Scalar,
)
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("ScalarToDiagonalSR2")
class ScalarToDiagonalSR2(Model):
    """Create a diagonal symmetric rank 2 tensor with values filled by a scalar."""

    # Forward: ``y = s * I`` where ``I`` is the SR2 identity. Linear leaf
    #: the directional pushforward along a Scalar tangent V is just
    # ``V * I`` -- the same typed wrapper map applied to the tangent. No
    # Jacobian materialised; ``SR2.identity`` and the typed ``Scalar * SR2``
    # operator already handle Mandel packing and sub-batch alignment, so the
    # body is pure wrapper algebra with no ``torch.<op>(.data)`` calls.

    hit = HitSchema(
        input("input", Scalar, "Scalar to convert"),
        output("output", SR2, "Output diagonal symmetric second order tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        s: Scalar,
        *promoted_params,
        v: ChainRuleDict | None = None,
    ):
        I = SR2.identity(dtype=s.data.dtype, device=s.data.device)
        y = s * I
        if v is None:
            return y

        def s_action(V: Scalar) -> SR2:
            return V * I

        return y, self.apply_chain_rule(v, "output", {"input": s_action}, output=y)


__all__ = ["ScalarToDiagonalSR2"]
