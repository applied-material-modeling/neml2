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

"""Subgrain coarsening rate, a native crystal-plasticity leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, option, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SubgrainCoarsening")
class SubgrainCoarsening(Model):
    r"""Diffusional coarsening of a subgrain structure, defined by
    $\dot{\lambda} = \dfrac{\mu b^3 K}{k T} \dfrac{1}{\lambda}$
    where $\lambda$ is the subgrain size, $\mu$ is the shear modulus,
    $b$ is the Burgers vector, $K$ is the coarsening coefficient,
    $k$ is the Boltzmann constant and $T$ is the absolute temperature.

    $K$ is a promotable parameter, so it may be supplied as a constant or
    driven by another model -- an ``ArrheniusParameter`` for the usual
    thermally activated form.
    """

    hit = HitSchema(
        input("subgrain_size", Scalar, "Current subgrain size"),
        input("temperature", Scalar, "Absolute temperature"),
        derived_output("subgrain_size", Scalar, attr="_lambda_rate", suffix="_rate"),
        parameter("coarsening_coefficient", Scalar, "Coarsening coefficient", allow_promotion=True),
        parameter("shear_modulus", Scalar, "Shear modulus", allow_promotion=True),
        parameter(
            "burgers_vector", Scalar, "Magnitude of the Burgers vector", allow_promotion=True
        ),
        option(
            "boltzmann_constant",
            float,
            "The Boltzmann constant, in units consistent with the other parameters",
            attr="k",
        ),
    )

    # ``from_hit`` auto-declares the three parameters, the ``k`` option and the
    # derived ``subgrain_size_rate`` output name (stored on ``self._lambda_rate``).
    coarsening_coefficient: Scalar
    shear_modulus: Scalar
    burgers_vector: Scalar
    k: float
    _lambda_rate: str

    def forward(  # type: ignore[override]
        self,
        lambda_sg: Scalar,
        temperature: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        K = self._get_param("coarsening_coefficient", promoted_params, Scalar)
        mu = self._get_param("shear_modulus", promoted_params, Scalar)
        b = self._get_param("burgers_vector", promoted_params, Scalar)
        k = self.k  # float option, not a registered parameter

        # numerator = mu * b^3 * K / k, so lambda_dot = numerator / (T * lambda)
        numerator = mu * b * b * b * K / k
        lambda_dot = numerator / temperature / lambda_sg

        if v is None:
            return lambda_dot

        # Differential pushforward. Closed-form coefficients of typed Scalars;
        # no Jacobian materialised.
        #   d lambda_dot / d lambda = -lambda_dot / lambda
        #   d lambda_dot / d T      = -lambda_dot / T
        #   d lambda_dot / d K      =  lambda_dot / K
        #   d lambda_dot / d mu     =  lambda_dot / mu
        #   d lambda_dot / d b      =  3 * lambda_dot / b
        actions: dict[str, ChainRuleAction] = {
            "subgrain_size": lambda V, c=-lambda_dot / lambda_sg: c * V,
            "temperature": lambda V, c=-lambda_dot / temperature: c * V,
        }
        if "coarsening_coefficient" in self._promoted_params:
            actions[self._promoted_params["coarsening_coefficient"].input_name] = (
                lambda V, c=lambda_dot / K: c * V
            )
        if "shear_modulus" in self._promoted_params:
            actions[self._promoted_params["shear_modulus"].input_name] = (
                lambda V, c=lambda_dot / mu: c * V
            )
        if "burgers_vector" in self._promoted_params:
            actions[self._promoted_params["burgers_vector"].input_name] = (
                lambda V, c=3.0 * lambda_dot / b: c * V
            )

        return lambda_dot, self.apply_chain_rule(v, self._lambda_rate, actions, output=lambda_dot)


__all__ = ["SubgrainCoarsening"]
