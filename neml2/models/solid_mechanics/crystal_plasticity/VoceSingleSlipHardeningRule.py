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

"""Python-native mirror of the C++ ``VoceSingleSlipHardeningRule`` crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("VoceSingleSlipHardeningRule")
class VoceSingleSlipHardeningRule(Model):
    r"""Voce hardening for a SingleSlipStrength type model defined by
    $\dot{\tau} = \theta_0 \left( 1 - \frac{\tau}{\tau_f} \right) \sum_{i=1}^{n_{slip}} \left| \dot{\gamma}_i \right|$
    where
    $\theta_0$ is the initial rate of work hardening, $\tau_f$ is
    the saturated, maximum value of the slip system strength, and
    $\dot{\gamma}_i$ is the slip rate on each system.
    """  # noqa: E501

    hit = HitSchema(
        input("slip_hardening", Scalar, "Name of current values of slip hardening"),
        input("sum_slip_rates", Scalar, "Name of tensor containing the sum of the slip rates"),
        output("slip_hardening_rate", Scalar, "Name of the slip hardening rate"),
        parameter("initial_slope", Scalar, "The initial rate of hardening", allow_promotion=True),
        parameter(
            "saturated_hardening",
            Scalar,
            "The final, saturated value of the slip system strength",
            allow_promotion=True,
        ),
    )

    # Both parameters are auto-declared by ``from_hit`` — no __init__ needed.
    initial_slope: Scalar
    saturated_hardening: Scalar

    def forward(  # type: ignore[override]
        self,
        tau: Scalar,
        sg: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        theta = self._get_param("initial_slope", promoted_params, Scalar)
        tauf = self._get_param("saturated_hardening", promoted_params, Scalar)
        out = theta * (1.0 - tau / tauf) * sg
        if v is None:
            return out
        # ∂out/∂tau = -theta/tauf · sg, ∂out/∂sg = theta(1 - tau/tauf) — typed Scalars.
        d_dtau = -theta / tauf * sg
        d_dsg = theta * (1.0 - tau / tauf)

        def tau_action(V: Scalar, c: Scalar = d_dtau) -> Scalar:
            return c * V

        def sg_action(V: Scalar, c: Scalar = d_dsg) -> Scalar:
            return c * V

        return out, self.apply_chain_rule(
            v,
            "slip_hardening_rate",
            {"slip_hardening": tau_action, "sum_slip_rates": sg_action},
            output=out,
        )
