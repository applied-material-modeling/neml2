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

"""Python-native mirror of the C++ ``ChabochePlasticHardening`` model."""

from __future__ import annotations

import torch

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, parameter
from ....types import SR2, Scalar, inner, log, norm
from ....types import pow as wpow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("ChabochePlasticHardening")
class ChabochePlasticHardening(Model):
    r"""Map the flow rate (i.e., the consistency parameter in the KKT conditions)
    to the rate of internal variables. This object defines the non-associative
    Fredrick-Armstrong kinematic hardening. In the model, back stress is
    directly treated as an internal variable. Rate of back stress is given as
    $\dot{\boldsymbol{X}} = \left( \frac{2}{3} C \frac{\partial f}{\partial \boldsymbol{M}} - g \boldsymbol{X} \right) \dot{\gamma}$.$\frac{\partial f}{\partial \boldsymbol{M}}$
    is the flow direction, $\dot{\gamma}$ is the flow rate, and $C$ and $g$ are material
    parameters. The complete Chaboche model adds static recovery terms
    $- A \lVert \boldsymbol{X} \rVert^{a - 1} \boldsymbol{X}$, so the model
    includes kinematic hardening, dynamic recovery, and static recovery. $A$ and $a$ are additional
    material parameters.
    """  # noqa: E501

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/plasticity/FredrickArmstrongPlasticHardening.h``
    # / ``ChabochePlasticHardening.h``); HIT renames cascade through the
    # schema's option-name == canonical-key convention.
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
            allow_nonlinear=True,
        ),
        parameter(
            "g",
            Scalar,
            "Dynamic recovery coefficient",
            attr="g",
            allow_nonlinear=True,
        ),
        parameter(
            "A",
            Scalar,
            "Static recovery prefactor",
            attr="A",
            allow_nonlinear=True,
        ),
        parameter(
            "a",
            Scalar,
            "Static recovery exponent",
            attr="a",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the four parameters and the derived
    # ``back_stress_rate`` output name (stored on ``self._X_rate``).
    C: Scalar
    g: Scalar
    A: Scalar
    a: Scalar
    _X_rate: str

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        flow_direction: SR2,
        back_stress: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Mirrors ``ChabochePlasticHardening::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/ChabochePlasticHardening.cxx``.
        C = self._get_param("C", nl_params, Scalar)
        g = self._get_param("g", nl_params, Scalar)
        A = self._get_param("A", nl_params, Scalar)
        a = self._get_param("a", nl_params, Scalar)

        gamma_dot = flow_rate
        NM = flow_direction
        X = back_stress

        eps = torch.finfo(X.dtype).eps  # machine_precision(dtype) regularizer
        s = norm(X, eps)
        # Part proportional to the plastic strain rate (kinematic hardening +
        # dynamic recovery).
        g_term = (2.0 / 3.0) * C * NM - g * X
        # Static recovery term.
        s_term = (-A) * wpow(s, a - 1) * X
        X_dot = g_term * gamma_dot + s_term

        if v is None:
            return X_dot

        # Differential pushforward. Each action takes a typed tangent V
        # of the input's type and returns the SR2 tangent of X_dot. No
        # Jacobian is materialised; ``inner(X, V)`` is a Scalar contraction
        # (X : V in Mandel packing) and ``Scalar * SR2`` broadcasts.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   d X_dot / d gamma_dot = g_term
        #   d X_dot / d NM        = (2/3) * C * gamma_dot * I_SR2
        #   d X_dot / d X         = -g*gamma_dot*I - A*pow(s, a-3)*((a-1) X:V X
        #                                            + s^2 V)
        #                         = -g*gamma_dot*V - A*pow(s, a-1)*V
        #                                          - A*(a-1)*pow(s, a-3)*(X:V)*X
        #   d X_dot / d C         = (2/3) * NM * gamma_dot
        #   d X_dot / d g         = -X * gamma_dot
        #   d X_dot / d A         = -pow(s, a-1) * X
        #   d X_dot / d a         = -A * X * pow(s, a-1) * log(s)
        s_am1 = wpow(s, a - 1)
        s_am3 = wpow(s, a - 3)
        coef_X_chain = A * (a - 1) * s_am3
        coef_X_lin = g * gamma_dot + A * s_am1

        actions = {
            "flow_rate": lambda V, c=g_term: c * V,
            "flow_direction": lambda V, c=(2.0 / 3.0) * C * gamma_dot: c * V,
            "back_stress": lambda V, c=coef_X_lin, k=coef_X_chain, Xc=X: (
                -(c * V) - k * inner(Xc, V) * Xc
            ),
        }
        # Nonlinear-parameter promotions: each parameter that was promoted to a
        # runtime input gets its own action keyed on the resolved input name.
        if "C" in self._nl_params:
            actions[self._nl_params["C"].input_name] = lambda V, c=(2.0 / 3.0) * NM * gamma_dot: (
                c * V
            )
        if "g" in self._nl_params:
            actions[self._nl_params["g"].input_name] = lambda V, c=-X * gamma_dot: c * V
        if "A" in self._nl_params:
            actions[self._nl_params["A"].input_name] = lambda V, c=-s_am1 * X: c * V
        if "a" in self._nl_params:
            a_coef = -A * X * s_am1 * log(s)
            actions[self._nl_params["a"].input_name] = lambda V, c=a_coef: c * V

        return X_dot, self.apply_chain_rule(v, self._X_rate, actions, output=X_dot)
