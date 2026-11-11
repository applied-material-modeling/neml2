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
"""Kocks-Mecking dynamic recovery coefficient, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, option, output, parameter
from ....types import Scalar, exp, gt, log, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("KocksMeckingDynamicRecovery")
class KocksMeckingDynamicRecovery(Model):
    r"""Kocks-Mecking rate- and temperature-dependent dynamic recovery
    coefficient,
    $k_2 = k_{2,0} \left( \dfrac{\max(\dot{\gamma}_\Sigma, \dot{\gamma}_{min})}{\dot{\gamma}_0} \right)^{-1/n}$
    where $\dot{\gamma}_\Sigma$ is the sum of the absolute slip rates,
    $\dot{\gamma}_{min}$ is a rate floor, $\dot{\gamma}_0$ is a reference
    rate, $k_{2,0}$ is the reference recovery coefficient and $n$ is the
    rate sensitivity.

    Recovery weakens as the slip rate rises and strengthens with
    temperature. Feed $n$ from ``KocksMeckingRateSensitivity`` (with $A < 0$,
    so $n > 0$) and $\dot{\gamma}_\Sigma$ from ``SumSlipRates``; the output is
    meant to drive the promoted ``k2`` of ``LumpedForestDislocationEvolution``.
    Use the physical (uncapped) $n$ here: a cap meant to protect a power-law
    slip rule would bias recovery at low temperature. Because
    $\dot{\gamma}_\Sigma$ sums over slip systems, $\dot{\gamma}_0$ is a
    reference slip-rate sum, larger than a macroscopic strain rate by roughly
    the Taylor factor.

    The floor keeps the logarithm finite when nothing slips; below it $k_2$
    is constant in the slip rate.
    """  # noqa: E501

    hit = HitSchema(
        input("sum_slip_rates", Scalar, "Sum of the absolute values of the slip rates"),
        input("rate_sensitivity", Scalar, "Rate sensitivity n; must be strictly positive"),
        output("k2", Scalar, "Dynamic recovery coefficient"),
        parameter("k20", Scalar, "Reference dynamic recovery coefficient", allow_promotion=True),
        parameter("gamma0", Scalar, "Reference slip-rate sum", allow_promotion=True),
        option(
            "rate_floor",
            float,
            "Slip-rate sum below which the rate dependence is frozen",
            default=1.0e-10,
            attr="_floor",
        ),
    )

    # ``from_hit`` auto-declares the two parameters; ``rate_floor`` lands on
    # ``self._floor`` via ``attr=``.
    k20: Scalar
    gamma0: Scalar
    _floor: float

    def forward(  # type: ignore[override]
        self,
        gamma_sum: Scalar,
        n: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        k20 = self._get_param("k20", promoted_params, Scalar)
        g0 = self._get_param("gamma0", promoted_params, Scalar)

        # max(gamma_sum, floor) as an explicit mask, so the tangent below can
        # thread the same mask and agree with autograd of the primal.
        active = gt(gamma_sum, self._floor)
        g_eff = where(active, gamma_sum, Scalar.from_value(self._floor, like=gamma_sum))
        log_ratio = log(g_eff / g0)
        factor = exp(-log_ratio / n)  # (g_eff / g0)^(-1/n)
        k2 = k20 * factor

        if v is None:
            return k2

        # Differential pushforward. Closed-form coefficients of typed Scalars:
        #   d k2 / d gamma_sum = -k2 / (n gamma_sum)   above the floor, 0 below
        #   d k2 / d n         =  k2 ln(g_eff / g0) / n^2
        #   d k2 / d k20       =  (g_eff / g0)^(-1/n)    (product form; k20 = 0 safe)
        #   d k2 / d gamma0    =  k2 / (n gamma0)
        d_dg = -k2 / (n * g_eff)

        def g_action(V: Scalar) -> Scalar:
            return where(active, d_dg * V, Scalar.zeros_like(V))

        actions: dict[str, ChainRuleAction] = {
            "sum_slip_rates": g_action,
            "rate_sensitivity": lambda V, c=k2 * log_ratio / (n * n): c * V,
        }
        promoted = self._promoted_params
        if "k20" in promoted:
            actions[promoted["k20"].input_name] = lambda V, c=factor: c * V
        if "gamma0" in promoted:
            actions[promoted["gamma0"].input_name] = lambda V, c=k2 / (n * g0): c * V

        return k2, self.apply_chain_rule(v, "k2", actions, output=k2)


__all__ = ["KocksMeckingDynamicRecovery"]
