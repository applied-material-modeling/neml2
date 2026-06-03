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

"""Python-native mirror of the C++ ``BurgersElement`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, derived_output, input, output, parameter
from ....types import SR2, Scalar


@register_native("BurgersElement")
class BurgersElement(Model):
    r"""Burgers viscoelastic model: a Maxwell element in series with a Kelvin-Voigt
    element. The shared stress is
    $\boldsymbol{\sigma} = E_M (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{v,M} - \boldsymbol{\varepsilon}_K)$,
    and the internal strains evolve as
    $\dot{\boldsymbol{\varepsilon}}_{v,M} = \boldsymbol{\sigma}/\eta_M$ and
    $\dot{\boldsymbol{\varepsilon}}_K = (\boldsymbol{\sigma} - E_K \boldsymbol{\varepsilon}_K) / \eta_K$.
    """  # noqa: E501

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/viscoelasticity/BurgersElement.h``);
    # the two ``*_rate`` output names are derived from the corresponding viscous
    # strain options via the schema's ``suffix="_rate"`` machinery, mirroring
    # the C++ ``rate_name(_EvM.name())`` / ``rate_name(_EK.name())`` convention.
    hit = HitSchema(
        input("strain", SR2, "Total strain"),
        input(
            "maxwell_viscous_strain",
            SR2,
            "Viscous strain in the Maxwell branch dashpot",
        ),
        input("kelvin_voigt_strain", SR2, "Strain in the Kelvin-Voigt branch"),
        output(
            "stress",
            SR2,
            "Total stress (shared between Maxwell and Kelvin-Voigt elements)",
        ),
        derived_output("maxwell_viscous_strain", SR2, attr="_EvM_rate", suffix="_rate"),
        derived_output("kelvin_voigt_strain", SR2, attr="_EK_rate", suffix="_rate"),
        parameter(
            "maxwell_modulus",
            Scalar,
            "Maxwell branch spring modulus",
            attr="EM",
            allow_nonlinear=True,
        ),
        parameter(
            "maxwell_viscosity",
            Scalar,
            "Maxwell branch dashpot viscosity",
            attr="etaM",
            allow_nonlinear=True,
        ),
        parameter(
            "kelvin_modulus",
            Scalar,
            "Kelvin-Voigt branch spring modulus",
            attr="EK",
            allow_nonlinear=True,
        ),
        parameter(
            "kelvin_viscosity",
            Scalar,
            "Kelvin-Voigt branch dashpot viscosity",
            attr="etaK",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the four parameters and the two derived
    # ``*_rate`` output names (stored on ``self._EvM_rate`` / ``self._EK_rate``).
    EM: Scalar
    etaM: Scalar
    EK: Scalar
    etaK: Scalar
    _EvM_rate: str
    _EK_rate: str

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        maxwell_viscous_strain: SR2,
        kelvin_voigt_strain: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Mirrors ``BurgersElement::set_value`` in
        # ``src/neml2/models/solid_mechanics/viscoelasticity/BurgersElement.cxx``.
        EM = self._get_param("EM", nl_params, Scalar)
        etaM = self._get_param("etaM", nl_params, Scalar)
        EK_p = self._get_param("EK", nl_params, Scalar)
        etaK = self._get_param("etaK", nl_params, Scalar)

        E = strain
        EvM = maxwell_viscous_strain
        EK_var = kelvin_voigt_strain

        Eel = E - EvM - EK_var
        S = EM * Eel
        EvM_dot = S / etaM
        EK_dot = (S - EK_p * EK_var) / etaK

        if v is None:
            return S, EvM_dot, EK_dot

        # Differential pushforward. The leaf is linear in every input
        # and parameter, so every action is a direct (Scalar or SR2) scaling of
        # the incoming tangent -- no Jacobian is materialised.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   d S       / d E      = EM * I_SR2
        #   d S       / d EvM    = -EM * I_SR2
        #   d S       / d EK     = -EM * I_SR2
        #   d EvM_dot / d E      = (EM / etaM) * I_SR2
        #   d EvM_dot / d EvM    = -(EM / etaM) * I_SR2
        #   d EvM_dot / d EK     = -(EM / etaM) * I_SR2
        #   d EK_dot  / d E      = (EM / etaK) * I_SR2
        #   d EK_dot  / d EvM    = -(EM / etaK) * I_SR2
        #   d EK_dot  / d EK     = -((EM + EK_p) / etaK) * I_SR2
        # Parameter-direction (when promoted to a nonlinear runtime input):
        #   d S       / d EM     = Eel
        #   d EvM_dot / d EM     = Eel / etaM
        #   d EK_dot  / d EM     = Eel / etaK
        #   d EvM_dot / d etaM   = -S / (etaM * etaM)
        #   d EK_dot  / d EK_p   = -EK_var / etaK
        #   d EK_dot  / d etaK   = -(S - EK_p * EK_var) / (etaK * etaK)
        EvM_dot_coef = EM / etaM
        EK_dot_coef_E = EM / etaK
        EK_dot_coef_EK = (EM + EK_p) / etaK

        # Stress actions (input directions).
        S_actions = {
            "strain": lambda V, c=EM: c * V,
            "maxwell_viscous_strain": lambda V, c=-EM: c * V,
            "kelvin_voigt_strain": lambda V, c=-EM: c * V,
        }
        # EvM_dot actions (input directions).
        EvM_actions = {
            "strain": lambda V, c=EvM_dot_coef: c * V,
            "maxwell_viscous_strain": lambda V, c=-EvM_dot_coef: c * V,
            "kelvin_voigt_strain": lambda V, c=-EvM_dot_coef: c * V,
        }
        # EK_dot actions (input directions).
        EK_actions = {
            "strain": lambda V, c=EK_dot_coef_E: c * V,
            "maxwell_viscous_strain": lambda V, c=-EK_dot_coef_E: c * V,
            "kelvin_voigt_strain": lambda V, c=-EK_dot_coef_EK: c * V,
        }

        # Nonlinear-parameter promotions: each parameter that was promoted to a
        # runtime input gets its own action on every output it affects, keyed by
        # the resolved input name.
        if "EM" in self._nl_params:
            name = self._nl_params["EM"].input_name
            S_actions[name] = lambda V, c=Eel: c * V
            EvM_actions[name] = lambda V, c=Eel / etaM: c * V
            EK_actions[name] = lambda V, c=Eel / etaK: c * V
        if "etaM" in self._nl_params:
            name = self._nl_params["etaM"].input_name
            EvM_actions[name] = lambda V, c=-S / (etaM * etaM): c * V
        if "EK" in self._nl_params:
            name = self._nl_params["EK"].input_name
            EK_actions[name] = lambda V, c=-EK_var / etaK: c * V
        if "etaK" in self._nl_params:
            name = self._nl_params["etaK"].input_name
            EK_actions[name] = lambda V, c=-(S - EK_p * EK_var) / (etaK * etaK): c * V

        v_S = self.apply_chain_rule(v, "stress", S_actions, output=S)
        v_EvM = self.apply_chain_rule(v, self._EvM_rate, EvM_actions, output=EvM_dot)
        v_EK = self.apply_chain_rule(v, self._EK_rate, EK_actions, output=EK_dot)
        # Merge per-output dicts (different outer keys).
        return S, EvM_dot, EK_dot, {**v_S, **v_EvM, **v_EK}
