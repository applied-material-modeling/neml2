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

"""Subgrain size to slip strength, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SubgrainStrength")
class SubgrainStrength(Model):
    r"""Strength contribution of a subgrain structure, defined by
    $\tau = \alpha \dfrac{\mu b}{\lambda}$
    where $\lambda$ is the subgrain size, $\alpha$ is an interaction
    coefficient, $\mu$ is the shear modulus and $b$ is the Burgers vector.

    The output is a single strength contribution, intended to be combined with
    the other contributions through a ``ScalarLinearCombination``.
    """

    hit = HitSchema(
        input("subgrain_size", Scalar, "Current subgrain size"),
        output("subgrain_strength", Scalar, "The subgrain strength contribution"),
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
        lambda_sg: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        alpha = self._get_param("alpha", promoted_params, Scalar)
        mu = self._get_param("shear_modulus", promoted_params, Scalar)
        b = self._get_param("burgers_vector", promoted_params, Scalar)

        tau = alpha * mu * b / lambda_sg

        if v is None:
            return tau

        # Differential pushforward. Closed-form coefficients of typed Scalars.
        #   d tau / d lambda = -tau / lambda
        #   d tau / d alpha  =  tau / alpha
        #   d tau / d mu     =  tau / mu
        #   d tau / d b      =  tau / b
        actions: dict[str, ChainRuleAction] = {
            "subgrain_size": lambda V, c=-tau / lambda_sg: c * V,
        }
        if "alpha" in self._promoted_params:
            actions[self._promoted_params["alpha"].input_name] = lambda V, c=tau / alpha: c * V
        if "shear_modulus" in self._promoted_params:
            actions[self._promoted_params["shear_modulus"].input_name] = lambda V, c=tau / mu: c * V
        if "burgers_vector" in self._promoted_params:
            actions[self._promoted_params["burgers_vector"].input_name] = lambda V, c=tau / b: c * V

        return tau, self.apply_chain_rule(v, "subgrain_strength", actions, output=tau)


__all__ = ["SubgrainStrength"]
