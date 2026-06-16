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

"""Python-native mirror of the C++ ``SumSlipRates`` crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import Scalar, sum
from ....types import abs as tensor_abs
from ....types import sign as tensor_sign
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("SumSlipRates")
class SumSlipRates(Model):
    r"""Calculates the sum of the absolute value of all the slip rates as
    $\sum_{i=1}^{n_{slip}} \left| \dot{\gamma}_i \right|$.
    """

    hit = HitSchema(
        input("slip_rates", Scalar, "The name of individual slip rates"),
        output("sum_slip_rates", Scalar, "The output name for the scalar sum of the slip rates"),
    )

    def forward(  # type: ignore[override]
        self,
        g: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # The slip axis is structurally the trailing dim of `g.data`. The
        # sub_batch_ndim hint resets to 0 across ComposedModel boundaries
        # , so we infer the reduction from tensor shape, not the hint.
        out = sum(tensor_abs(g).sub_batch, -1)
        if v is None:
            return out
        # ∂(Σ|γ_k|)/∂γ_k = sign(γ_k); pushforward weights the per-slip tangent
        # then sums over slip — pure typed Scalar algebra.
        sign_g = tensor_sign(g)

        def g_action(V: Scalar) -> Scalar:
            return sum((sign_g * V).sub_batch, -1)

        return out, self.apply_chain_rule(v, "sum_slip_rates", {"slip_rates": g_action}, output=out)
