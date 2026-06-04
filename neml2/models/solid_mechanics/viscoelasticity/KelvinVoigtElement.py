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

"""Python-native mirror of the C++ ``KelvinVoigtElement`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import SR2, Scalar


@register_neml2_object("KelvinVoigtElement")
class KelvinVoigtElement(Model):
    r"""Stress response of a Kelvin-Voigt viscoelastic element (spring and
    dashpot in parallel),
    $\boldsymbol{\sigma} = E \boldsymbol{\varepsilon} + \eta \dot{\boldsymbol{\varepsilon}}$, where
    $E$ is the spring
    modulus and $\eta$ is the dashpot viscosity.
    """

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/viscoelasticity/KelvinVoigtElement.h``);
    # the ``strain_rate`` input name is derived from the ``strain`` option
    # via the schema's ``suffix="_rate"`` machinery, mirroring the C++
    # ``rate_name(_E.name())`` convention.
    hit = HitSchema(
        input("strain", SR2, "Strain shared by the spring and the dashpot"),
        derived_input("strain", SR2, attr="_E_dot", suffix="_rate"),
        output("stress", SR2, "Stress in the Kelvin-Voigt element"),
        parameter(
            "modulus",
            Scalar,
            "Spring modulus",
            attr="K",
            allow_nonlinear=True,
        ),
        parameter(
            "viscosity",
            Scalar,
            "Dashpot viscosity",
            attr="eta",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the two parameters and the derived
    # ``strain_rate`` input name (stored on ``self._E_dot``).
    K: Scalar
    eta: Scalar
    _E_dot: str

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        strain_rate: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Mirrors ``KelvinVoigtElement::set_value`` in
        # ``src/neml2/models/solid_mechanics/viscoelasticity/KelvinVoigtElement.cxx``.
        K = self._get_param("K", nl_params, Scalar)
        eta = self._get_param("eta", nl_params, Scalar)
        E = strain
        E_dot = strain_rate
        S = K * E + eta * E_dot
        if v is None:
            return S
        # Differential pushforward. The forward is linear in every
        # argument, so each action just scales the incoming tangent by the
        # appropriate (Scalar or SR2) coefficient -- no Jacobian is materialised.
        #   d S / d E      = K * I_SR2        -> K * V
        #   d S / d E_dot  = eta * I_SR2      -> eta * V
        #   d S / d K      = E                -> E * V          (V Scalar)
        #   d S / d eta    = E_dot            -> E_dot * V      (V Scalar)
        actions = {
            "strain": lambda V, c=K: c * V,
            self._E_dot: lambda V, c=eta: c * V,
        }
        if "K" in self._nl_params:
            actions[self._nl_params["K"].input_name] = lambda V, c=E: c * V
        if "eta" in self._nl_params:
            actions[self._nl_params["eta"].input_name] = lambda V, c=E_dot: c * V
        return S, self.apply_chain_rule(v, "stress", actions, output=S)
