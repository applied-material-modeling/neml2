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

"""Python-native mirror of the C++ ``FredrickArmstrongPlasticHardening`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import SR2, Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("FredrickArmstrongPlasticHardening")
class FredrickArmstrongPlasticHardening(Model):
    r"""Map the flow rate (i.e., the consistency parameter in the KKT conditions)
    to the rate of internal variables. This object defines the non-associative
    Fredrick-Armstrong kinematic hardening. In the model, back stress is
    directly treated as an internal variable. Rate of back stress is given as
    $\dot{\boldsymbol{X}} = \left( \frac{2}{3} C \frac{\partial f}{\partial \boldsymbol{M}} - g \boldsymbol{X} \right) \dot{\gamma}$.$\frac{\partial f}{\partial \boldsymbol{M}}$
    is the flow direction, $\dot{\gamma}$ is the flow rate, and $C$ and $g$ are material
    parameters.
    """  # noqa: E501

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/plasticity/FredrickArmstrongPlasticHardening.h``);
    # HIT renames cascade through the schema's option-name == canonical-key
    # convention.
    hit = HitSchema(
        input("flow_rate", Scalar, "Flow rate"),
        input("flow_direction", SR2, "Flow direction"),
        input("back_stress", SR2, "Back stress"),
        # ``back_stress_rate`` is derived from the ``back_stress`` option,
        # matching the C++ ``rate_name(_X.name())`` convention.
        derived_output("back_stress", SR2, attr="_X_rate", suffix="_rate"),
        parameter(
            "C",
            Scalar,
            "Kinematic hardening coefficient",
            attr="C",
            allow_promotion=True,
        ),
        parameter(
            "g",
            Scalar,
            "Dynamic recovery coefficient",
            attr="g",
            allow_promotion=True,
        ),
    )

    # ``from_hit`` auto-declares the two parameters and the derived
    # ``back_stress_rate`` output name (stored on ``self._X_rate``).
    C: Scalar
    g: Scalar
    _X_rate: str

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        flow_direction: SR2,
        back_stress: SR2,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Mirrors ``FredrickArmstrongPlasticHardening::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/FredrickArmstrongPlasticHardening.cxx``.
        C = self._get_param("C", promoted_params, Scalar)
        g = self._get_param("g", promoted_params, Scalar)

        gamma_dot = flow_rate
        NM = flow_direction
        X = back_stress

        # Part proportional to the plastic strain rate (kinematic hardening +
        # dynamic recovery).
        g_term = (2.0 / 3.0) * C * NM - g * X
        X_dot = g_term * gamma_dot

        if v is None:
            return X_dot

        # Differential pushforward. Each action takes a typed tangent V
        # of the input's type and returns the SR2 tangent of X_dot. Linear leaf
        # in (gamma_dot, NM, X, C, g) -- no Jacobian is materialised.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   d X_dot / d gamma_dot = g_term
        #   d X_dot / d NM        = (2/3) * C * gamma_dot * I_SR2
        #   d X_dot / d X         = -g * gamma_dot * I_SR2
        #   d X_dot / d C         = (2/3) * NM * gamma_dot
        #   d X_dot / d g         = -X * gamma_dot
        actions = {
            "flow_rate": lambda V, c=g_term: c * V,
            "flow_direction": lambda V, c=(2.0 / 3.0) * C * gamma_dot: c * V,
            "back_stress": lambda V, c=-g * gamma_dot: c * V,
        }
        # Promoted-parameter contributions: each parameter that was promoted to a
        # runtime input gets its own action keyed on the resolved input name.
        if "C" in self._promoted_params:
            actions[self._promoted_params["C"].input_name] = (
                lambda V, c=(2.0 / 3.0) * NM * gamma_dot: c * V
            )
        if "g" in self._promoted_params:
            actions[self._promoted_params["g"].input_name] = lambda V, c=-X * gamma_dot: c * V

        return X_dot, self.apply_chain_rule(v, self._X_rate, actions, output=X_dot)
