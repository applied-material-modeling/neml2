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

"""Python-native mirror of the C++ ``ArrheniusParameter`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import BLOCK_NAME, HitSchema, input, option, output, parameter
from ...types import Scalar, exp
from ..chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ..model import Model


@register_neml2_object("ArrheniusParameter")
class ArrheniusParameter(Model):
    r"""Define the variable as a function of temperature according to the
    Arrhenius law $p = p_0 \exp \left( -\frac{Q}{RT} \right)$, where
    $p_0$ is the reference value, $Q$ is the activation energy,
    $R$ is the ideal gas constant, and $T$ is the temperature.
    """

    # Smooth in T (T > 0) ⇒ second-order chain rule is well-defined.
    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input("temperature", Scalar, "Temperature"),
        output(
            "parameter",
            Scalar,
            "The output parameter. If not specified, the object name will be used.",
            default=BLOCK_NAME,
        ),
        parameter("reference_value", Scalar, "Reference value", attr="p0"),
        parameter("activation_energy", Scalar, "Activation energy", attr="Q"),
        option("ideal_gas_constant", float, "The ideal gas constant", attr="R"),
    )

    # ``from_hit`` auto-declares ``reference_value`` (stored as ``p0``) and
    # ``activation_energy`` (stored as ``Q``); the ``R`` option lands on
    # ``self`` via ``attr=``. Annotate so pyright sees the typed wrappers that
    # ``Model.__getattr__`` returns rather than ``nn.Module``'s ``Module`` hint.
    p0: Scalar
    Q: Scalar
    R: float

    def forward(  # type: ignore[override]
        self,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        T = temperature
        p0 = self.p0
        Q = self.Q
        R = self.R
        p = p0 * exp(-Q / R / T)
        if v is None:
            return p
        # Differential pushforward: ∂p/∂T = p · Q / (R · T²) — a Scalar
        # coefficient scaling the incoming Scalar tangent. No Jacobian formed.
        dp_dT = p * Q / R / T / T

        def temperature_action(V: Scalar) -> Scalar:
            return dp_dT * V

        actions_1 = {"temperature": temperature_action}

        # Second-order Hessian: d²p/dT² = p · Q · (Q - 2 R T) / (R² T⁴).
        # Derivation: d/dT [p · Q / (R T²)] = (dp/dT)·Q/(R T²) + p·Q·(-2/(R T³))
        #           = p·Q²/(R² T⁴) - 2·p·Q/(R T³)
        #           = p·Q·(Q - 2 R T) / (R² T⁴).
        d2p_dT2 = p * Q * (Q - 2.0 * R * T) / (R * R * T * T * T * T)

        def temperature_temperature_action(Va: Scalar, Vb: Scalar) -> Scalar:
            return d2p_dT2 * Va * Vb

        actions_2 = {("temperature", "temperature"): temperature_temperature_action}
        return p, *self.propagate_tangents(
            v, "parameter", actions_1, output=p, v2=v2, actions_2=actions_2, vh=vh
        )


__all__ = ["ArrheniusParameter"]
