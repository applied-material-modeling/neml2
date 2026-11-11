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

"""Lumped (single-density) forest dislocation evolution, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, sqrt
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("LumpedForestDislocationEvolution")
class LumpedForestDislocationEvolution(Model):
    r"""Standard forest hardening model for a single, lumped dislocation
    density shared by all slip systems, defined by
    $\dot{\rho} = (k_1 \sqrt{\rho} - k_2 \rho) \sum_{i=1}^{n_{slip}} \left| \dot{\gamma}_i \right|$.

    This is the lumped counterpart of ``PerSlipForestDislocationEvolution``:
    the density carries no slip-system axis, and it is driven by the total
    slip rate (typically from ``SumSlipRates``) rather than by the slip rate
    on each system. Paired with ``DislocationObstacleStrengthMap``, it gives
    the same Taylor strength on every slip system.
    """  # noqa: E501

    # The output ``dislocation_density_rate`` is derived from the
    # ``dislocation_density`` input, matching PerSlipForestDislocationEvolution.
    hit = HitSchema(
        input("dislocation_density", Scalar, "Lumped dislocation density"),
        input("sum_slip_rates", Scalar, "Sum of the absolute values of the slip rates"),
        derived_output("dislocation_density", Scalar, attr="_rho_rate", suffix="_rate"),
        parameter("k1", Scalar, "Hardening coefficient", attr="k1", allow_promotion=True),
        parameter("k2", Scalar, "Recovery coefficient", attr="k2", allow_promotion=True),
    )

    # ``from_hit`` auto-declares the two parameters and the derived
    # ``dislocation_density_rate`` output name (stored on ``self._rho_rate``).
    k1: Scalar
    k2: Scalar
    _rho_rate: str

    def forward(  # type: ignore[override]
        self,
        rho: Scalar,
        gamma_sum: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        k1 = self._get_param("k1", promoted_params, Scalar)
        k2 = self._get_param("k2", promoted_params, Scalar)

        sqrt_rho = sqrt(rho)
        hardening = k1 * sqrt_rho - k2 * rho
        rho_dot = hardening * gamma_sum

        if v is None:
            return rho_dot

        # Differential pushforward. Closed-form coefficients of typed
        # Scalars; no Jacobian materialised. The input is already the sum of
        # absolute slip rates, so rho_dot is linear in it:
        #   d rho_dot / d rho       = (k1 * 0.5 / sqrt(rho) - k2) * sum|gamma_dot|
        #   d rho_dot / d sum       = hardening
        #   d rho_dot / d k1        = sqrt(rho) * sum|gamma_dot|
        #   d rho_dot / d k2        = -rho * sum|gamma_dot|
        d_drho = (0.5 * k1 / sqrt_rho - k2) * gamma_sum

        actions = {
            "dislocation_density": lambda V, c=d_drho: c * V,
            "sum_slip_rates": lambda V, c=hardening: c * V,
        }
        if "k1" in self._promoted_params:
            actions[self._promoted_params["k1"].input_name] = lambda V, c=sqrt_rho * gamma_sum: (
                c * V
            )
        if "k2" in self._promoted_params:
            actions[self._promoted_params["k2"].input_name] = lambda V, c=-rho * gamma_sum: c * V

        return rho_dot, self.apply_chain_rule(v, self._rho_rate, actions, output=rho_dot)


__all__ = ["LumpedForestDislocationEvolution"]
