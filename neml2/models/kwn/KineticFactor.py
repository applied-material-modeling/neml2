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

"""Python-native mirror of the C++ ``KineticFactor`` model."""

from __future__ import annotations

import math

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, pow


@register_native("KineticFactor")
class KineticFactor(Model):
    r"""Nucleation kinetic factor.

    Computes :math:`\beta = 4\pi \, (N_a / V_m)^{4/3} \, R_{\mathrm{crit}}^2 /
    \mathrm{sum}`, where :math:`R_{\mathrm{crit}}` is the critical radius for
    nucleation, :math:`\mathrm{sum}` is the projected diffusivity sum,
    :math:`V_m` is the molar volume of the precipitate, and :math:`N_a` is
    Avogadro's number.
    """

    hit = HitSchema(
        input("critical_radius", Scalar, "Critical radius for nucleation"),
        input("projected_diffusivity_sum", Scalar, "Projected diffusivity sum"),
        output("kinetic_factor", Scalar, "Kinetic factor for nucleation"),
        parameter(
            "molar_volume",
            Scalar,
            "Molar volume of the precipitate",
            attr="V_m",
            allow_nonlinear=True,
        ),
        parameter(
            "avogadro_number",
            Scalar,
            "Avogadro's number",
            attr="N_a",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``molar_volume`` parameter (stored as
    # ``V_m``) and the ``avogadro_number`` parameter (stored as ``N_a``).
    # Annotate so pyright sees the typed wrappers that ``Model.__getattr__``
    # returns.
    V_m: Scalar
    N_a: Scalar

    def forward(  # type: ignore[override]
        self,
        critical_radius: Scalar,
        projected_diffusivity_sum: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        R = critical_radius
        s = projected_diffusivity_sum
        V_m = self._get_param("V_m", nl_params, Scalar)
        N_a = self._get_param("N_a", nl_params, Scalar)

        # Forward: β = 4π (N_a / V_m)^(4/3) R² / sum. Typed Scalar algebra
        # end-to-end, matching ``KineticFactor::set_value``. ``pow`` is
        # declared generic over ``TensorWrapper``; narrow the result back to
        # ``Scalar`` so pyright sees Scalar algebra in the products that
        # follow.
        coef = (4.0 * math.pi) * pow(N_a, 4.0 / 3.0) / pow(V_m, 4.0 / 3.0)
        beta = coef * R * R / s

        if v is None:
            return beta

        # Differential pushforward. Linear coefficients:
        #   ∂β/∂R    = +2β / R          ⇒ action(V) = (2β/R) · V
        #   ∂β/∂sum  = -β  / sum        ⇒ action(V) = (-β/sum) · V
        #   ∂β/∂V_m  = -(4/3) β / V_m   ⇒ action(V) = (-(4/3)β/V_m) · V
        #   ∂β/∂N_a  = +(4/3) β / N_a   ⇒ action(V) = (+(4/3)β/N_a) · V
        # The parameter actions only fire when the parameter has been promoted
        # to a nonlinear input via the HIT ``[Models]`` cross-ref form.
        dbeta_dR = 2.0 * beta / R
        dbeta_ds = -beta / s
        actions: dict[str, ChainRuleAction] = {
            "critical_radius": lambda V, c=dbeta_dR: c * V,
            "projected_diffusivity_sum": lambda V, c=dbeta_ds: c * V,
        }

        V_m_nlp = self._nl_params.get("V_m")
        if V_m_nlp is not None:
            dbeta_dVm = -(4.0 / 3.0) * beta / V_m
            actions[V_m_nlp.input_name] = lambda V, c=dbeta_dVm: c * V

        N_a_nlp = self._nl_params.get("N_a")
        if N_a_nlp is not None:
            dbeta_dNa = (4.0 / 3.0) * beta / N_a
            actions[N_a_nlp.input_name] = lambda V, c=dbeta_dNa: c * V

        return beta, self.apply_chain_rule(v, "kinetic_factor", actions, output=beta)


__all__ = ["KineticFactor"]
