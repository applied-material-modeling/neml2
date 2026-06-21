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

"""Python-native mirror of the C++ ``ZeldovichFactor`` model."""

from __future__ import annotations

import math

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, sqrt
from ..chain_rule import ChainRuleAction, ChainRuleDict
from ..model import Model


@register_neml2_object("ZeldovichFactor")
class ZeldovichFactor(Model):
    r"""Zeldovich factor for nucleation.

    Computes $Z = \dfrac{V_m}{2\pi N_a R_{\mathrm{crit}}^2}
    \sqrt{\dfrac{\gamma}{k T}}$, where $R_{\mathrm{crit}}$ is the
    critical radius for nucleation, $\gamma$ is the surface energy of the
    precipitate, $T$ is the temperature, $V_m$ is the molar volume,
    $N_a$ is Avogadro's number, and $k$ is the Boltzmann constant.
    """

    hit = HitSchema(
        input("critical_radius", Scalar, "Critical radius for nucleation"),
        input("temperature", Scalar, "Temperature"),
        output("zeldovich_factor", Scalar, "Zeldovich factor for nucleation"),
        parameter(
            "surface_energy",
            Scalar,
            "Surface energy of the precipitate",
            attr="gamma",
            allow_nonlinear=True,
        ),
        parameter(
            "molar_volume",
            Scalar,
            "Molar volume of the precipitate",
            attr="V_m",
        ),
        parameter(
            "avogadro_number",
            Scalar,
            "Avogadro's number",
            attr="N_a",
        ),
        parameter(
            "boltzmann_constant",
            Scalar,
            "Boltzmann constant",
            attr="k",
        ),
    )

    # ``from_hit`` auto-declares the four parameters above (stored under their
    # ``attr`` names). Annotate so pyright sees the typed wrappers that
    # ``Model.__getattr__`` returns.
    gamma: Scalar
    V_m: Scalar
    N_a: Scalar
    k: Scalar

    def forward(  # type: ignore[override]
        self,
        critical_radius: Scalar,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        R = critical_radius
        T = temperature
        gamma = self._get_param("gamma", nl_params, Scalar)
        V_m = self._get_param("V_m", nl_params, Scalar)
        N_a = self._get_param("N_a", nl_params, Scalar)
        k = self._get_param("k", nl_params, Scalar)

        # Forward: Z = V_m / (2π N_a R²) · sqrt(γ / (k T)). Typed Scalar
        # algebra end-to-end, matching ``ZeldovichFactor::set_value``.
        coef = V_m / ((2.0 * math.pi) * N_a * R * R)
        root = sqrt(gamma / (k * T))
        Z = coef * root

        if v is None:
            return Z

        # Differential pushforward. Linear coefficients:
        #   ∂Z/∂R     = -2Z / R       ⇒ action(V) = (-2Z/R) · V
        #   ∂Z/∂T     = -Z / (2T)     ⇒ action(V) = (-Z/(2T)) · V
        #   ∂Z/∂gamma = +Z / (2γ)     ⇒ action(V) = (+Z/(2γ)) · V
        # The gamma action only fires when the parameter has been promoted to a
        # nonlinear input via the HIT ``[Models]`` cross-ref form. V_m, N_a,
        # and k are not declared ``allow_nonlinear`` (matching the C++).
        dZ_dR = -2.0 * Z / R
        dZ_dT = -0.5 * Z / T
        actions: dict[str, ChainRuleAction] = {
            "critical_radius": lambda V, c=dZ_dR: c * V,
            "temperature": lambda V, c=dZ_dT: c * V,
        }

        gamma_nlp = self._nl_params.get("gamma")
        if gamma_nlp is not None:
            dZ_dgamma = 0.5 * Z / gamma
            actions[gamma_nlp.input_name] = lambda V, c=dZ_dgamma: c * V

        return Z, self.apply_chain_rule(v, "zeldovich_factor", actions, output=Z)


__all__ = ["ZeldovichFactor"]
