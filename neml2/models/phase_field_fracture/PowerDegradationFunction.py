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

"""Python-native mirror of the C++ ``PowerDegradationFunction`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar
from ...types.functions import pow as type_pow


@register_neml2_object("PowerDegradationFunction")
class PowerDegradationFunction(Model):
    r"""Power degradation function used to degrade the elastic strain energy
    density,
    $g = \left( 1 - d \right)^p (1 - \eta) + \eta$,
    where $d$ is the phase-field variable, $p$ the power, and
    $\eta$ a residual degradation when $d = 1$.
    """

    # Smooth in d (for p ≥ 2) ⇒ second-order chain rule is well-defined.
    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input("phase", Scalar, "Phase-field variable"),
        output("degradation", Scalar, "Value of the degradation function"),
        parameter("power", Scalar, "Power of the degradation function", attr="p"),
        option("eta", float, "Residual degradation when d = 1", default=0.0, attr="_eta"),
    )

    p: Scalar
    _eta: float

    def forward(  # type: ignore[override]
        self,
        phase: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        p = self._get_param("p", nl_params, Scalar)
        eta = self._eta
        one_minus_d = 1.0 - phase
        # g = (1 - d)**p * (1 - eta) + eta
        g = type_pow(one_minus_d, p) * (1.0 - eta) + eta
        if v is None:
            return g

        # First-order: dg/dd = -p * (1 - d)**(p - 1) * (1 - eta).
        dgdd = -p * type_pow(one_minus_d, p - 1.0) * (1.0 - eta)

        def phase_action(V: Scalar) -> Scalar:
            return dgdd * V

        actions_1 = {"phase": phase_action}

        # Second-order Hessian: d^2g / dd^2 = p (p - 1) (1 - d)^(p - 2) (1 - eta).
        d2gdd2 = p * (p - 1.0) * type_pow(one_minus_d, p - 2.0) * (1.0 - eta)

        def phase_phase_action(Va: Scalar, Vb: Scalar) -> Scalar:
            return d2gdd2 * Va * Vb

        actions_2 = {("phase", "phase"): phase_phase_action}
        return g, *self.propagate_tangents(
            v, "degradation", actions_1, output=g, v2=v2, actions_2=actions_2, vh=vh
        )
