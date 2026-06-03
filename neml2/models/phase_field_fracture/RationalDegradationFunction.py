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

"""Python-native mirror of the C++ ``RationalDegradationFunction`` model."""

from __future__ import annotations

from typing import cast

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar
from ...types.functions import pow as type_pow


@register_native("RationalDegradationFunction")
class RationalDegradationFunction(Model):
    r"""Rational degradation function used to degrade the elastic strain energy
    density,
    $g = \frac{(1-d)^p}{(1-d)^p + Q(d)} (1 - \eta) + \eta$,
    where $Q(d) = b_1 d (1 + b_2 d + b_2 b_3 d^2)$, $d$ is the
    phase-field variable, $p$ the power, $\eta$ a residual
    degradation when $d = 1$, and $b_1, b_2, b_3$ material fitting
    parameters.
    """

    hit = HitSchema(
        input("phase", Scalar, "Phase-field variable"),
        output("degradation", Scalar, "Value of the degradation function"),
        parameter("power", Scalar, "Power of the degradation function", attr="p"),
        parameter("b1", Scalar, "Degradation parameter b_1", attr="b1"),
        parameter("b2", Scalar, "Degradation parameter b_2", attr="b2"),
        parameter("b3", Scalar, "Degradation parameter b_3", attr="b3"),
        option("eta", float, "Residual degradation when d = 1", default=0.0, attr="_eta"),
    )

    # ``from_hit`` auto-declares the parameters (stored under their ``attr``);
    # the ``eta`` option lands on ``self._eta`` via ``attr=``. Annotate so
    # pyright sees the typed wrapper returned by ``Model.__getattr__``.
    p: Scalar
    b1: Scalar
    b2: Scalar
    b3: Scalar
    _eta: float

    def forward(  # type: ignore[override]
        self,
        phase: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        p = self._get_param("p", nl_params, Scalar)
        b1 = self._get_param("b1", nl_params, Scalar)
        b2 = self._get_param("b2", nl_params, Scalar)
        b3 = self._get_param("b3", nl_params, Scalar)
        eta = self._eta

        d = phase
        one_minus_d = 1.0 - d
        # f = (1 - d)^p ; Q = b1 d (1 + b2 d + b2 b3 d^2)
        f = cast(Scalar, type_pow(one_minus_d, p))
        Q = b1 * d * (1.0 + b2 * d + b2 * b3 * d * d)
        D = f + Q
        # g = (f / D) (1 - eta) + eta
        g = (f / D) * (1.0 - eta) + eta
        if v is None:
            return g

        # D-062 pushforward: write the analytical scalar derivative dg/dd in
        # typed wrapper algebra. Quotient rule on g = (f/D)(1-eta) + eta gives
        #   dg/dd = (f' D - f D') / D^2 (1 - eta)
        # Since D = f + Q, D' = f' + Q', so the numerator collapses to
        #   f' D - f D' = f' (f + Q) - f (f' + Q') = f' Q - f Q'.
        # With f' = -p (1-d)^(p-1) and Q' = b1 (1 + 2 b2 d + 3 b2 b3 d^2).
        f_prime = -p * cast(Scalar, type_pow(one_minus_d, p - 1.0))
        Q_prime = b1 * (1.0 + 2.0 * b2 * d + 3.0 * b2 * b3 * d * d)
        dgdd = (f_prime * Q - f * Q_prime) / (D * D) * (1.0 - eta)

        def phase_action(V: Scalar) -> Scalar:
            return dgdd * V

        actions = {"phase": phase_action}
        return g, self.apply_chain_rule(v, "degradation", actions, output=g)
