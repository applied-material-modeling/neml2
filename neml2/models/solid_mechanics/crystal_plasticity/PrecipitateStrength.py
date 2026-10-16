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

"""Precipitate population to slip strength, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("PrecipitateStrength")
class PrecipitateStrength(Model):
    r"""Strength contribution of a precipitate population, defined by
    $\tau = \alpha \dfrac{\mu b}{2 R N}$
    where $R$ is the mean precipitate radius, $N$ is the precipitate
    number density, $\alpha$ is an interaction coefficient, $\mu$ is the
    shear modulus and $b$ is the Burgers vector.

    The output is a single strength contribution, intended to be combined with
    the other contributions through a ``ScalarLinearCombination``.
    """

    hit = HitSchema(
        input("precipitate_radius", Scalar, "Mean precipitate radius"),
        input("precipitate_density", Scalar, "Precipitate number density"),
        output("precipitate_strength", Scalar, "The precipitate strength contribution"),
        parameter("alpha", Scalar, "Interaction coefficient", allow_promotion=True),
        parameter("shear_modulus", Scalar, "Shear modulus", allow_promotion=True),
        parameter(
            "burgers_vector", Scalar, "Magnitude of the Burgers vector", allow_promotion=True
        ),
    )

    # ``from_hit`` auto-declares the three parameters via the schema.
    alpha: Scalar
    shear_modulus: Scalar
    burgers_vector: Scalar

    def forward(  # type: ignore[override]
        self,
        R: Scalar,
        N: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        alpha = self._get_param("alpha", promoted_params, Scalar)
        mu = self._get_param("shear_modulus", promoted_params, Scalar)
        b = self._get_param("burgers_vector", promoted_params, Scalar)

        two_RN = 2.0 * R * N
        tau = alpha * mu * b / two_RN

        if v is None:
            return tau

        # Differential pushforward. Closed-form coefficients of typed Scalars.
        #   d tau / d R     = -tau / R
        #   d tau / d N     = -tau / N
        #   d tau / d alpha =  mu b / (2 R N)
        #   d tau / d mu    =  alpha b / (2 R N)
        #   d tau / d b     =  alpha mu / (2 R N)
        # The three parameter coefficients differentiate the product directly
        # rather than forming ``tau / <param>``: zeroing a coefficient is a
        # valid way to switch this contribution off, and the quotient form is
        # 0/0 there.
        actions: dict[str, ChainRuleAction] = {
            "precipitate_radius": lambda V, c=-tau / R: c * V,
            "precipitate_density": lambda V, c=-tau / N: c * V,
        }
        if "alpha" in self._promoted_params:
            actions[self._promoted_params["alpha"].input_name] = lambda V, c=mu * b / two_RN: c * V
        if "shear_modulus" in self._promoted_params:
            actions[self._promoted_params["shear_modulus"].input_name] = (
                lambda V, c=alpha * b / two_RN: c * V
            )
        if "burgers_vector" in self._promoted_params:
            actions[self._promoted_params["burgers_vector"].input_name] = (
                lambda V, c=alpha * mu / two_RN: c * V
            )

        return tau, self.apply_chain_rule(v, "precipitate_strength", actions, output=tau)


__all__ = ["PrecipitateStrength"]
