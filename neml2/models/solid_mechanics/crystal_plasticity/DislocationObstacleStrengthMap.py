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

"""Python-native mirror of the C++ ``DislocationObstacleStrengthMap`` leaf."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sqrt
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("DislocationObstacleStrengthMap")
class DislocationObstacleStrengthMap(Model):
    r"""Dislocation density to strength as in Taylor
    $\tau_i = \tau_{const} + \alpha \mu b \sqrt{\rho_i}$.
    """

    # Variable / parameter names match the C++ defaults so the same .i
    # scenarios drive both backends.
    hit = HitSchema(
        input("dislocation_density", Scalar, "Per-slip dislocation density"),
        output("slip_strengths", Scalar, "Name of the slip system strengths"),
        parameter("constant_strength", Scalar, "Constant strength offset", allow_nonlinear=True),
        parameter("alpha", Scalar, "Interaction coefficient", allow_nonlinear=True),
        parameter("mu", Scalar, "Shear modulus", allow_nonlinear=True),
        parameter("b", Scalar, "Burgers vector", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the four parameters via the schema.
    constant_strength: Scalar
    alpha: Scalar
    mu: Scalar
    b: Scalar

    def forward(  # type: ignore[override]
        self,
        rho: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``DislocationObstacleStrengthMap::set_value`` in the C++ source:
        #   tau = tau_const + alpha * mu * b * sqrt(rho)
        tau_const = self._get_param("constant_strength", nl_params, Scalar)
        alpha = self._get_param("alpha", nl_params, Scalar)
        mu = self._get_param("mu", nl_params, Scalar)
        b = self._get_param("b", nl_params, Scalar)

        sqrt_rho = sqrt(rho)
        coeff = alpha * mu * b
        tau = tau_const + coeff * sqrt_rho

        if v is None:
            return tau

        # Differential pushforward. Closed-form coefficients of typed
        # Scalars; no Jacobian materialised. Derivation matches the dense
        # C++ jacobian rows in ``set_value``:
        #   d tau / d rho               = coeff * 0.5 / sqrt(rho)
        #   d tau / d constant_strength = 1
        #   d tau / d alpha             = mu * b * sqrt(rho)
        #   d tau / d mu                = alpha * b * sqrt(rho)
        #   d tau / d b                 = alpha * mu * sqrt(rho)
        d_drho = coeff * 0.5 / sqrt_rho

        actions: dict[str, ChainRuleAction] = {
            "dislocation_density": lambda V, c=d_drho: c * V,
        }
        if "constant_strength" in self._nl_params:
            actions[self._nl_params["constant_strength"].input_name] = lambda V: V
        if "alpha" in self._nl_params:
            actions[self._nl_params["alpha"].input_name] = lambda V, c=mu * b * sqrt_rho: c * V
        if "mu" in self._nl_params:
            actions[self._nl_params["mu"].input_name] = lambda V, c=alpha * b * sqrt_rho: c * V
        if "b" in self._nl_params:
            actions[self._nl_params["b"].input_name] = lambda V, c=alpha * mu * sqrt_rho: c * V

        return tau, self.apply_chain_rule(v, "slip_strengths", actions, output=tau)
