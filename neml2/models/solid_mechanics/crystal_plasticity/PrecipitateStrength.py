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
from ....types import Scalar, sqrt
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("PrecipitateStrength")
class PrecipitateStrength(Model):
    r"""Orowan (bypass) strength contribution of a precipitate population,
    defined by
    $\tau = \alpha \mu b \sqrt{2 R N}$
    where $R$ is the mean precipitate radius, $N$ is the precipitate
    number density (per unit volume), $\alpha$ is an interaction
    coefficient, $\mu$ is the shear modulus and $b$ is the Burgers vector.

    A slip plane intersects every precipitate whose center lies within $R$
    of it, so the number of obstacles per unit area of slip plane is
    $2 R N$ and their spacing is $L = 1 / \sqrt{2 R N}$; the strength is
    the Orowan stress $\alpha \mu b / L$. At fixed volume fraction
    $f = \tfrac{4}{3} \pi R^3 N$ this is $\tau \propto \sqrt{f} / R$, so the
    contribution falls as the population coarsens (overaging).

    Both $R$ and $N$ must be strictly positive: the derivatives of the square
    root are unbounded where $R N = 0$.

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

        # Obstacle density per unit slip-plane area is 2 R N; its square root
        # is the inverse obstacle spacing 1/L.
        inv_L = sqrt(2.0 * R * N)
        tau = alpha * mu * b * inv_L

        if v is None:
            return tau

        # Differential pushforward. Closed-form coefficients of typed Scalars.
        #   d tau / d R     = tau / (2 R)
        #   d tau / d N     = tau / (2 N)
        #   d tau / d alpha = mu b sqrt(2 R N)
        #   d tau / d mu    = alpha b sqrt(2 R N)
        #   d tau / d b     = alpha mu sqrt(2 R N)
        # The three parameter coefficients differentiate the product directly
        # rather than forming ``tau / <param>``: zeroing a coefficient is a
        # valid way to switch this contribution off, and the quotient form is
        # 0/0 there.
        actions: dict[str, ChainRuleAction] = {
            "precipitate_radius": lambda V, c=0.5 * tau / R: c * V,
            "precipitate_density": lambda V, c=0.5 * tau / N: c * V,
        }
        if "alpha" in self._promoted_params:
            actions[self._promoted_params["alpha"].input_name] = lambda V, c=mu * b * inv_L: c * V
        if "shear_modulus" in self._promoted_params:
            actions[self._promoted_params["shear_modulus"].input_name] = (
                lambda V, c=alpha * b * inv_L: c * V
            )
        if "burgers_vector" in self._promoted_params:
            actions[self._promoted_params["burgers_vector"].input_name] = (
                lambda V, c=alpha * mu * inv_L: c * V
            )

        return tau, self.apply_chain_rule(v, "precipitate_strength", actions, output=tau)


__all__ = ["PrecipitateStrength"]
