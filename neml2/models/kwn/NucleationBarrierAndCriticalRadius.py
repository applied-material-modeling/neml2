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

"""Python-native mirror of the C++ ``NucleationBarrierAndCriticalRadius`` model."""

from __future__ import annotations

import math

from ...factory import register_neml2_object
from ...schema import HitSchema, output, parameter
from ...types import Scalar
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("NucleationBarrierAndCriticalRadius")
class NucleationBarrierAndCriticalRadius(Model):
    r"""Compute the nucleation critical radius and Gibbs free energy barrier.

    Mirrors the C++ ``NucleationBarrierAndCriticalRadius``:

    $$
    R_{\mathrm{crit}} = \dfrac{2\,\gamma\,V_m}{\Delta g_{\mathrm{total}}},
    \qquad
    \Delta g = \dfrac{16}{3}\,\pi\,\gamma^{3}\,V_m^{2}
               \big/ \Delta g_{\mathrm{total}}^{2},
    $$

    where $\gamma$ is the surface energy of the precipitate,
    $V_m$ is the molar volume, and $\Delta g_{\mathrm{total}}$
    is the total Gibbs free energy difference driving nucleation.
    """

    hit = HitSchema(
        output(
            "nucleation_barrier",
            Scalar,
            "Gibbs free energy barrier for nucleation",
        ),
        output(
            "critical_radius",
            Scalar,
            "Critical radius for nucleation",
        ),
        parameter(
            "surface_energy",
            Scalar,
            "Surface energy of the precipitate",
            attr="gamma",
            allow_promotion=True,
        ),
        parameter(
            "total_gibbs_free_energy_difference",
            Scalar,
            "Total Gibbs free energy difference driving nucleation",
            attr="dg_total",
            allow_promotion=True,
        ),
        parameter(
            "molar_volume",
            Scalar,
            "Molar volume of the precipitate",
            attr="V_m",
        ),
    )

    # ``from_hit`` auto-declares the three parameters above (stored under their
    # ``attr`` names). Annotate so pyright sees the typed wrappers that
    # ``Model.__getattr__`` returns.
    gamma: Scalar
    dg_total: Scalar
    V_m: Scalar

    def forward(  # type: ignore[override]
        self,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> tuple[Scalar, Scalar] | tuple[Scalar, Scalar, ChainRuleDict]:
        # The model has no structural inputs: ``*promoted_params`` only carries the
        # optional promoted parameters (``gamma`` and/or ``dg_total`` in
        # mode 3/4). ``V_m`` is not declared ``allow_promotion`` (matching the
        # C++ ctor), so it always stays static.
        gamma = self._get_param("gamma", promoted_params, Scalar)
        dg_total = self._get_param("dg_total", promoted_params, Scalar)
        V_m = self._get_param("V_m", promoted_params, Scalar)

        # Forward: R_crit = 2 gamma V_m / dg_total
        #          dg     = (16/3) pi gamma^3 V_m^2 / dg_total^2
        # Typed Scalar algebra end-to-end, matching
        # ``NucleationBarrierAndCriticalRadius::set_value``.
        R_crit = (2.0 * gamma * V_m) / dg_total
        gamma2 = gamma * gamma
        V_m2 = V_m * V_m
        dg_total2 = dg_total * dg_total
        dg = ((16.0 / 3.0) * math.pi) * (gamma2 * gamma) * V_m2 / dg_total2

        if v is None:
            return dg, R_crit

        # Differential pushforward. Linear coefficients (mirroring the
        # C++ ``set_value`` partials):
        #   ∂dg/∂gamma     = 3 dg / gamma         ⇒ action(V) = (3·dg/gamma)·V
        #   ∂dg/∂dg_total  = -2 dg / dg_total     ⇒ action(V) = (-2·dg/dg_total)·V
        #   ∂R_crit/∂gamma    = 2 V_m / dg_total  ⇒ action(V) = (2·V_m/dg_total)·V
        #   ∂R_crit/∂dg_total = -R_crit/dg_total  ⇒ action(V) = (-R_crit/dg_total)·V
        # Each parameter's action only fires when it has been promoted to a
        # runtime input via the HIT ``[Models]`` cross-ref form. V_m is not
        # ``allow_promotion`` (matching the C++).
        dg_actions: dict[str, ChainRuleAction] = {}
        R_actions: dict[str, ChainRuleAction] = {}

        gamma_nlp = self._promoted_params.get("gamma")
        if gamma_nlp is not None:
            ddg_dgamma = 3.0 * dg / gamma
            dR_dgamma = (2.0 * V_m) / dg_total
            dg_actions[gamma_nlp.input_name] = lambda V, c=ddg_dgamma: c * V
            R_actions[gamma_nlp.input_name] = lambda V, c=dR_dgamma: c * V

        dg_total_nlp = self._promoted_params.get("dg_total")
        if dg_total_nlp is not None:
            ddg_ddgt = -2.0 * dg / dg_total
            dR_ddgt = -R_crit / dg_total
            dg_actions[dg_total_nlp.input_name] = lambda V, c=ddg_ddgt: c * V
            R_actions[dg_total_nlp.input_name] = lambda V, c=dR_ddgt: c * V

        v_dg = self.apply_chain_rule(v, "nucleation_barrier", dg_actions, output=dg)
        v_R = self.apply_chain_rule(v, "critical_radius", R_actions, output=R_crit)
        # Merge per-output dicts (different outer keys).
        return dg, R_crit, {**v_dg, **v_R}


__all__ = ["NucleationBarrierAndCriticalRadius"]
