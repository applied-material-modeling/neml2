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

"""Python-native mirror of the C++ ``SlopeSaturationVoceIsotropicHardening`` model."""

from __future__ import annotations

from typing import cast

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, sign


@register_native("SlopeSaturationVoceIsotropicHardening")
class SlopeSaturationVoceIsotropicHardening(Model):
    r"""SlopeSaturationVoce isotropic hardening model,
    $\dot{h} = \theta_0 \left(1 - \frac{h}{R} \right) \varepsilon_p$, where $R$ is the
    isotropic hardening upon saturation, and $\theta_0$ is the initial
    hardening rate. In addition to the reparameterization, this model
    differences from the `VoceIsotropicHardening` model in that it defines the
    hardening rate in a non-assocative manner.  This is sometimes handy, for
    example in supplementing the model with static recovery.
    """

    # Variable names match the C++ defaults in FlowRule (flow_rate) and the
    # declared input ``isotropic_hardening``; HIT renames cascade through the
    # schema's option-name == canonical-key convention. The rate output is
    # derived from the ``isotropic_hardening`` input name with the ``_rate``
    # suffix — when HIT renames the input (e.g. ``isotropic_hardening =
    # 'k_voce'``) the rate output follows automatically (``k_voce_rate``).
    # Mirrors C++ ``declare_output_variable<Scalar>(rate_name(_h.name()))``.
    hit = HitSchema(
        input("flow_rate", Scalar, "Flow rate"),
        input("isotropic_hardening", Scalar, "Isotropic hardening variable"),
        derived_output("isotropic_hardening", Scalar, attr="_h_rate", suffix="_rate"),
        parameter(
            "saturated_hardening",
            Scalar,
            "Saturated isotropic hardening",
            attr="R",
            allow_nonlinear=True,
        ),
        parameter(
            "initial_hardening_rate",
            Scalar,
            "Initial hardening rate",
            attr="theta0",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``saturated_hardening`` /
    # ``initial_hardening_rate`` parameters (stored as ``R`` / ``theta0``) -- no
    # __init__ needed.
    R: Scalar
    theta0: Scalar
    _h_rate: str

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        isotropic_hardening: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``SlopeSaturationVoceIsotropicHardening::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/SlopeSaturationVoceIsotropicHardening.cxx``.
        gamma_dot = flow_rate
        h = isotropic_hardening
        R = self._get_param("R", nl_params, Scalar)
        theta0 = self._get_param("theta0", nl_params, Scalar)

        # h_dot = sign(R) * theta0 * (1 - h / R) * gamma_dot
        sR = cast(Scalar, sign(R))
        one_minus_h_over_R = -(h / R) + 1.0
        h_dot = sR * theta0 * one_minus_h_over_R * gamma_dot

        if v is None:
            return h_dot

        # Differential pushforward. Each action takes a typed Scalar
        # tangent of the input's type and returns the Scalar tangent of h_dot.
        # No Jacobian is materialised; each coefficient is a Scalar that
        # broadcasts against the leading-K tangent axis automatically.
        #
        # Derivations match the dense C++ Jacobian in set_value:
        #   d h_dot / d gamma_dot = sign(R) * theta0 * (1 - h / R)
        #   d h_dot / d h         = -sign(R) * theta0 / R * gamma_dot
        #   d h_dot / d R         = gamma_dot * h * sign(R) * theta0 / (R * R)
        #   d h_dot / d theta0    = sign(R) * (1 - h / R) * gamma_dot
        coef_gamma_dot = sR * theta0 * one_minus_h_over_R
        coef_h = -sR * theta0 / R * gamma_dot

        actions = {
            "flow_rate": lambda V, c=coef_gamma_dot: c * V,
            "isotropic_hardening": lambda V, c=coef_h: c * V,
        }
        if "R" in self._nl_params:
            coef_R = gamma_dot * h * sR * theta0 / (R * R)
            actions[self._nl_params["R"].input_name] = lambda V, c=coef_R: c * V
        if "theta0" in self._nl_params:
            coef_theta0 = sR * one_minus_h_over_R * gamma_dot
            actions[self._nl_params["theta0"].input_name] = lambda V, c=coef_theta0: c * V

        return h_dot, self.apply_chain_rule(v, self._h_rate, actions, output=h_dot)
