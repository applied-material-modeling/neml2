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

"""Python-native mirror of the C++ ``PowerLawSlipRule`` crystal-plasticity leaf."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ....types import abs as tensor_abs
from ....types import pow as tensor_pow


@register_neml2_object("PowerLawSlipRule")
class PowerLawSlipRule(Model):
    r"""Power law slip rule defined as
    $\dot{\gamma}_i = \dot{\gamma}_0 \left| \frac{\tau_i}{\hat{\tau}_i} \right|^{n-1} \frac{\tau_i}{\hat{\tau}_i}$
    with $\dot{\gamma}_i$
    the slip rate on system $i$, $\tau_i$ the resolved shear,
    $\hat{\tau}_i$ the slip system strength, $n$ the rate
    senstivity, and $\dot{\gamma}_0$ a reference slip rate.
    """  # noqa: E501

    hit = HitSchema(
        input("resolved_shears", Scalar, "Name of the resolved shear tensor"),
        input("slip_strengths", Scalar, "Name of the tensor containing the slip system strengths"),
        output("slip_rates", Scalar, "Name of the slip rate tensor"),
        parameter("gamma0", Scalar, "Reference slip rate", allow_nonlinear=True),
        parameter("n", Scalar, "Rate sensitivity exponent", allow_nonlinear=True),
    )

    # Both parameters are auto-declared by ``from_hit`` — no __init__ needed.
    gamma0: Scalar
    n: Scalar

    def forward(  # type: ignore[override]
        self,
        rss: Scalar,
        tau: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        gamma0 = self._get_param("gamma0", nl_params, Scalar)
        nv = self._get_param("n", nl_params, Scalar)
        ratio = rss / tau
        abs_ratio = tensor_abs(ratio)
        g = gamma0 * tensor_pow(abs_ratio, nv - 1.0) * ratio
        if v is None:
            return g

        # Per-slip closed-form derivatives, as typed Scalar coefficients:
        #   ∂γ̇/∂τ = n γ0 |τ/τ̂|^(n−1) / τ̂
        #   ∂γ̇/∂τ̂ = -n γ0 τ |τ|^(n−1) / τ̂^(n+1)
        dg_drss = nv * gamma0 * tensor_pow(abs_ratio, nv - 1.0) / tau
        dg_dtau = -(
            nv * gamma0 * rss * tensor_pow(tensor_abs(rss), nv - 1.0) / tensor_pow(tau, nv + 1.0)
        )

        def rss_action(V: Scalar, c: Scalar = dg_drss) -> Scalar:
            return c * V

        def tau_action(V: Scalar, c: Scalar = dg_dtau) -> Scalar:
            return c * V

        return g, self.apply_chain_rule(
            v,
            "slip_rates",
            {"resolved_shears": rss_action, "slip_strengths": tau_action},
            output=g,
        )
