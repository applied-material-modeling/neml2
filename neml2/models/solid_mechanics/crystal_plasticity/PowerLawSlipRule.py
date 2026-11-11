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

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, log, where
from ....types import abs as tensor_abs
from ....types import opaque_pow as tensor_pow
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


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
        parameter("gamma0", Scalar, "Reference slip rate", allow_promotion=True),
        parameter("n", Scalar, "Rate sensitivity exponent", allow_promotion=True),
    )

    # Both parameters are auto-declared by ``from_hit`` — no __init__ needed.
    gamma0: Scalar
    n: Scalar

    def forward(  # type: ignore[override]
        self,
        rss: Scalar,
        tau: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        gamma0 = self._get_param("gamma0", promoted_params, Scalar)
        nv = self._get_param("n", promoted_params, Scalar)
        ratio = rss / tau
        abs_ratio = tensor_abs(ratio)
        # Factor the slip rate as ``g = f * ratio`` where ``f`` is the
        # magnitude part. ``tensor_pow`` is opaque to Inductor (routed
        # through ``neml2::opaque_pow``, see ``types/functions.py``), so
        # the pow stays as a single per-(B, slip) compute rather than
        # being inlined into the downstream K-batched per-slip reduction
        # kernel and recomputed K × n_comp times redundantly.
        pw = tensor_pow(abs_ratio, nv - 1.0)
        f = gamma0 * pw
        g = f * ratio
        if v is None:
            return g

        # Per-slip closed-form derivatives, in terms of ``f`` (no extra ``pow``):
        #
        #   ∂γ̇/∂τ   = n γ0 |τ/τ̂|^(n−1) / τ̂  = n · f / τ̂
        #   ∂γ̇/∂τ̂  = −n γ0 τ |τ|^(n−1) / τ̂^(n+1) = −n · f · (τ/τ̂) / τ̂ = −n · g / τ̂
        #
        # Algebraically identical to the naive expressions and well-defined
        # at ``τ → 0`` (no 0/0). Crucial under AOTI: Inductor fuses ``pow``
        # into the downstream K-batched per-slip reduction kernel and
        # computes it ~K × n_comp times redundantly (its inputs ``τ``,
        # ``τ̂`` don't depend on K or n_comp). Hoisting ``pow`` out of the
        # derivative expressions into the shared ``f`` lets the fused
        # kernel reuse the materialised ``f`` value instead -- standalone
        # microbench of the dominant scpcoup step kernel: 3.56 ms with
        # inline pow vs 0.22 ms with precomputed factor (16x).
        dg_drss = nv * f / tau
        dg_dtau = -nv * g / tau

        def rss_action(V: Scalar, c: Scalar = dg_drss) -> Scalar:
            return c * V

        def tau_action(V: Scalar, c: Scalar = dg_dtau) -> Scalar:
            return c * V

        actions: dict[str, ChainRuleAction] = {
            "resolved_shears": rss_action,
            "slip_strengths": tau_action,
        }
        promoted = self._promoted_params
        if "gamma0" in promoted:
            # dg/dgamma0 = |ratio|^(n-1) * ratio; product form, so gamma0 = 0
            # stays finite (the quotient g / gamma0 would be 0/0).
            actions[promoted["gamma0"].input_name] = lambda V, c=pw * ratio: c * V
        if "n" in promoted:
            # dg/dn = g * ln|ratio|. At ratio = 0 the limit is 0 (g vanishes
            # faster than the log diverges for n > 1); feeding log a safe base
            # of 1 there gives exactly that 0 instead of 0 * -inf = nan.
            nonzero = gt(abs_ratio, 0.0)
            safe_abs = where(nonzero, abs_ratio, Scalar.from_value(1.0, like=abs_ratio))
            dg_dn = g * log(safe_abs)
            actions[promoted["n"].input_name] = lambda V, c=dg_dn: c * V

        return g, self.apply_chain_rule(v, "slip_rates", actions, output=g)
