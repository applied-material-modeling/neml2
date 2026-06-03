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

"""Python-native mirror of the C++ ``LinearSingleSlipHardeningRule`` crystal-plasticity leaf."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar


@register_native("LinearSingleSlipHardeningRule")
class LinearSingleSlipHardeningRule(Model):
    r"""Simple linear slip system hardening defined by
    $\dot{\tau} = \theta \sum_{i=1}^{n_{slip}} \left| \dot{\gamma}_i \right|$
    where $\theta$ is the hardening slope.
    """

    hit = HitSchema(
        input("slip_hardening", Scalar, "Name of current values of slip hardening"),
        input("sum_slip_rates", Scalar, "Name of tensor containing the sum of the slip rates"),
        output("slip_hardening_rate", Scalar, "Name of the slip hardening rate"),
        parameter("hardening_slope", Scalar, "Hardening rate", allow_nonlinear=True),
    )

    # Auto-declared by ``from_hit`` — no __init__ needed.
    hardening_slope: Scalar

    def forward(  # type: ignore[override]
        self,
        tau: Scalar,
        sg: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        theta = self._get_param("hardening_slope", nl_params, Scalar)
        out = theta * sg
        if v is None:
            return out

        # Linear leaf in sg with constant slope theta; slip_hardening (tau) is unused
        # in the forward, so its action is structural zero (no entry registered).
        def sg_action(V: Scalar, c: Scalar = theta) -> Scalar:
            return c * V

        return out, self.apply_chain_rule(
            v,
            "slip_hardening_rate",
            {"sum_slip_rates": sg_action},
            output=out,
        )
