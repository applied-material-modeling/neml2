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

"""Python-native mirror of the C++ ``LinearDashpot`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import SR2, Scalar


@register_neml2_object("LinearDashpot")
class LinearDashpot(Model):
    r"""Newtonian dashpot constitutive law,
    $\dot{\boldsymbol{\varepsilon}} = \boldsymbol{\sigma} / \eta$, where $\eta$ is the viscosity.
    This
    is the leaf dashpot element used in any rheological network composition
    (Maxwell, Kelvin-Voigt, Zener, Wiechert, Burgers, and arbitrary
    user-defined topologies).
    """

    hit = HitSchema(
        input("stress", SR2, "Stress acting across the dashpot"),
        output(
            "viscous_strain_rate",
            SR2,
            "Rate of viscous strain. Override to match the rate name expected by the "
            "time integrator if your state variable is not named `viscous_strain`.",
        ),
        parameter(
            "viscosity",
            Scalar,
            "Dashpot viscosity",
            attr="eta",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``viscosity`` parameter (stored as
    # ``eta``) -- no __init__ needed.
    eta: Scalar

    def forward(  # type: ignore[override]
        self,
        stress: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        S = stress
        eta = self._get_param("eta", nl_params, Scalar)
        Ev_dot = S / eta
        if v is None:
            return Ev_dot
        # Differential pushforward: Ev_dot = S / eta. Linear in S, so
        # the S-direction action scales the SR2 tangent by 1/eta. The
        # eta-direction (when promoted to a nonlinear parameter input) is the
        # ordinary -S/eta^2 derivative scaling the incoming Scalar tangent.
        actions = {"stress": lambda V, c=eta: V / c}
        if "eta" in self._nl_params:
            actions[self._nl_params["eta"].input_name] = lambda V, c=eta, s=S: -(s / (c * c)) * V
        return Ev_dot, self.apply_chain_rule(v, "viscous_strain_rate", actions, output=Ev_dot)
