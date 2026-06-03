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

"""Python-native mirror of the C++ ``WiechertElement`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, derived_output, input, output, parameter
from ....types import SR2, Scalar


@register_native("WiechertElement")
class WiechertElement(Model):
    r"""Wiechert (generalized Maxwell) viscoelastic model with two Maxwell branches in
    parallel with an equilibrium spring. Total stress is
    $\boldsymbol{\sigma} = (E_\infty + E_1 + E_2) \boldsymbol{\varepsilon} - E_1 \boldsymbol{\varepsilon}_{v,1} - E_2 \boldsymbol{\varepsilon}_{v,2}$,
    with
    each viscous strain evolving according to
    $\dot{\boldsymbol{\varepsilon}}_{v,i} = E_i (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{v,i}) / \eta_i$.
    """  # noqa: E501

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/viscoelasticity/WiechertElement.h``);
    # the two ``*_rate`` output names are derived from the corresponding viscous
    # strain options via the schema's ``suffix="_rate"`` machinery, mirroring
    # the C++ ``rate_name(_Ev1.name())`` / ``rate_name(_Ev2.name())`` convention.
    hit = HitSchema(
        input("strain", SR2, "Total strain"),
        input("viscous_strain_1", SR2, "Viscous strain in the first Maxwell branch"),
        input("viscous_strain_2", SR2, "Viscous strain in the second Maxwell branch"),
        output("stress", SR2, "Total stress"),
        derived_output("viscous_strain_1", SR2, attr="_Ev1_rate", suffix="_rate"),
        derived_output("viscous_strain_2", SR2, attr="_Ev2_rate", suffix="_rate"),
        parameter(
            "equilibrium_modulus",
            Scalar,
            "Equilibrium spring modulus",
            attr="Einf",
            allow_nonlinear=True,
        ),
        parameter(
            "modulus_1",
            Scalar,
            "Spring modulus of the first Maxwell branch",
            attr="E1",
            allow_nonlinear=True,
        ),
        parameter(
            "viscosity_1",
            Scalar,
            "Dashpot viscosity of the first Maxwell branch",
            attr="eta1",
            allow_nonlinear=True,
        ),
        parameter(
            "modulus_2",
            Scalar,
            "Spring modulus of the second Maxwell branch",
            attr="E2",
            allow_nonlinear=True,
        ),
        parameter(
            "viscosity_2",
            Scalar,
            "Dashpot viscosity of the second Maxwell branch",
            attr="eta2",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the five parameters and the two derived
    # ``*_rate`` output names (stored on ``self._Ev1_rate`` / ``self._Ev2_rate``).
    Einf: Scalar
    E1: Scalar
    eta1: Scalar
    E2: Scalar
    eta2: Scalar
    _Ev1_rate: str
    _Ev2_rate: str

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        viscous_strain_1: SR2,
        viscous_strain_2: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Mirrors ``WiechertElement::set_value`` in
        # ``src/neml2/models/solid_mechanics/viscoelasticity/WiechertElement.cxx``.
        Einf = self._get_param("Einf", nl_params, Scalar)
        E1 = self._get_param("E1", nl_params, Scalar)
        eta1 = self._get_param("eta1", nl_params, Scalar)
        E2 = self._get_param("E2", nl_params, Scalar)
        eta2 = self._get_param("eta2", nl_params, Scalar)

        E = strain
        Ev1 = viscous_strain_1
        Ev2 = viscous_strain_2

        diff1 = E - Ev1
        diff2 = E - Ev2
        S1 = E1 * diff1
        S2 = E2 * diff2

        S = Einf * E + S1 + S2
        Ev1_dot = S1 / eta1
        Ev2_dot = S2 / eta2

        if v is None:
            return S, Ev1_dot, Ev2_dot

        # Differential pushforward. The leaf is linear in every input
        # and parameter, so every action is a direct (Scalar or SR2) scaling of
        # the incoming tangent -- no Jacobian is materialised.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   d S       / d E    = (Einf + E1 + E2) * I_SR2
        #   d S       / d Ev1  = -E1 * I_SR2
        #   d S       / d Ev2  = -E2 * I_SR2
        #   d Ev1_dot / d E    = (E1 / eta1) * I_SR2
        #   d Ev1_dot / d Ev1  = -(E1 / eta1) * I_SR2
        #   d Ev2_dot / d E    = (E2 / eta2) * I_SR2
        #   d Ev2_dot / d Ev2  = -(E2 / eta2) * I_SR2
        # Parameter-direction (when promoted to a nonlinear runtime input):
        #   d S       / d Einf = E
        #   d S       / d E1   = diff1
        #   d Ev1_dot / d E1   = diff1 / eta1
        #   d Ev1_dot / d eta1 = -S1 / (eta1 * eta1)
        #   d S       / d E2   = diff2
        #   d Ev2_dot / d E2   = diff2 / eta2
        #   d Ev2_dot / d eta2 = -S2 / (eta2 * eta2)
        S_coef_E = Einf + E1 + E2
        Ev1_dot_coef = E1 / eta1
        Ev2_dot_coef = E2 / eta2

        # Stress actions (input directions).
        S_actions = {
            "strain": lambda V, c=S_coef_E: c * V,
            "viscous_strain_1": lambda V, c=-E1: c * V,
            "viscous_strain_2": lambda V, c=-E2: c * V,
        }
        # Ev1_dot actions (input directions).
        Ev1_actions = {
            "strain": lambda V, c=Ev1_dot_coef: c * V,
            "viscous_strain_1": lambda V, c=-Ev1_dot_coef: c * V,
        }
        # Ev2_dot actions (input directions).
        Ev2_actions = {
            "strain": lambda V, c=Ev2_dot_coef: c * V,
            "viscous_strain_2": lambda V, c=-Ev2_dot_coef: c * V,
        }

        # Nonlinear-parameter promotions: each parameter that was promoted to a
        # runtime input gets its own action on every output it affects, keyed by
        # the resolved input name.
        if "Einf" in self._nl_params:
            name = self._nl_params["Einf"].input_name
            S_actions[name] = lambda V, c=E: c * V
        if "E1" in self._nl_params:
            name = self._nl_params["E1"].input_name
            S_actions[name] = lambda V, c=diff1: c * V
            Ev1_actions[name] = lambda V, c=diff1 / eta1: c * V
        if "eta1" in self._nl_params:
            name = self._nl_params["eta1"].input_name
            Ev1_actions[name] = lambda V, c=-S1 / (eta1 * eta1): c * V
        if "E2" in self._nl_params:
            name = self._nl_params["E2"].input_name
            S_actions[name] = lambda V, c=diff2: c * V
            Ev2_actions[name] = lambda V, c=diff2 / eta2: c * V
        if "eta2" in self._nl_params:
            name = self._nl_params["eta2"].input_name
            Ev2_actions[name] = lambda V, c=-S2 / (eta2 * eta2): c * V

        v_S = self.apply_chain_rule(v, "stress", S_actions, output=S)
        v_Ev1 = self.apply_chain_rule(v, self._Ev1_rate, Ev1_actions, output=Ev1_dot)
        v_Ev2 = self.apply_chain_rule(v, self._Ev2_rate, Ev2_actions, output=Ev2_dot)
        # Merge per-output dicts (different outer keys).
        return S, Ev1_dot, Ev2_dot, {**v_S, **v_Ev1, **v_Ev2}
