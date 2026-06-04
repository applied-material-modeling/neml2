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

"""Python-native mirrors of C++ ``common/DynamicMean.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import (
    SR2,
    mean,
)


@register_neml2_object("SR2DynamicMean")
class SR2DynamicMean(Model):
    """``to = mean(from, dim=dynamic_dim)``. Reduces a named dynamic batch dim.

    Mirrors C++ ``DynamicMean<SR2>``. The ``dim`` HIT option selects the
    dynamic-batch axis to average over (the same convention as the C++
    ``Size`` argument: 0 is the leading dynamic axis, ``-1`` the trailing).
    """

    # The reduction couples sites along the dynamic axis. From the chain rule's
    # perspective this isn't a sub-batch concern (sub-batch sparsity lives along
    # the trailing sub-batch axes); we leave list_deriv empty.
    hit = HitSchema(
        input("from", SR2, "Input tensor to average"),
        output("to", SR2, "Averaged tensor"),
        option("dim", int, "The dimension to average over", attr="_dim"),
    )
    _dim: int

    def forward(  # type: ignore[override]
        self,
        x: SR2,
        v: ChainRuleDict | None = None,
    ):
        d = self._dim
        out = mean(x.dynamic_batch, d)
        if v is None:
            return out

        # Mean along a dynamic axis is a uniform 1/N reduction; the pushforward
        # applies the same reduction to the tangent. Under the leading-K
        # convention K is prepended as the leading dynamic axis, so a
        # non-negative input-frame dim shifts right by one; a negative dim
        # (counted from the end of the dynamic region) is unaffected.
        d_tan = d + 1 if d >= 0 else d

        def action(V: SR2) -> SR2:
            return mean(V.dynamic_batch, d_tan)

        return out, self.apply_chain_rule(v, "to", {"from": action}, output=out)


__all__ = ["SR2DynamicMean"]
