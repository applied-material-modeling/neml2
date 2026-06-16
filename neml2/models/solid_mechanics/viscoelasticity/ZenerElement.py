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

"""Python-native mirror of the C++ ``ZenerElement`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_output, input, output, parameter
from ....types import SR2, Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("ZenerElement")
class ZenerElement(Model):
    r"""Zener (Standard Linear Solid) viscoelastic model: an equilibrium spring in
    parallel with a Maxwell branch. The total stress is
    $\boldsymbol{\sigma} = (E_\infty + E_M) \boldsymbol{\varepsilon} - E_M \boldsymbol{\varepsilon}_v$,
    and the Maxwell-branch viscous strain evolves as
    $\dot{\boldsymbol{\varepsilon}}_v = E_M (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_v) / \eta_M$.
    """  # noqa: E501

    # Variable names match the C++ defaults
    # (``include/neml2/models/solid_mechanics/viscoelasticity/ZenerElement.h``).
    # The ``viscous_strain_rate`` output name is derived from the
    # ``viscous_strain`` input option via the schema's ``suffix="_rate"``
    # machinery, mirroring the C++ ``rate_name(_Ev.name())`` convention.
    hit = HitSchema(
        input("strain", SR2, "Total strain"),
        input("viscous_strain", SR2, "Viscous strain in the Maxwell branch"),
        output("stress", SR2, "Total stress"),
        derived_output("viscous_strain", SR2, attr="_Ev_dot", suffix="_rate"),
        parameter(
            "equilibrium_modulus",
            Scalar,
            "Equilibrium spring modulus",
            attr="Einf",
            allow_nonlinear=True,
        ),
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
    )

    # ``from_hit`` auto-declares the three parameters and stores the resolved
    # derived ``viscous_strain_rate`` output name on ``self._Ev_dot``
    # (``derived_output`` does not register a rename entry, so the leaf must
    # pass the resolved name to ``apply_chain_rule`` directly).
    Einf: Scalar
    EM: Scalar
    etaM: Scalar
    _Ev_dot: str

    def forward(  # type: ignore[override]
        self,
        strain: SR2,
        viscous_strain: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Mirrors ``ZenerElement::set_value`` in
        # ``src/neml2/models/solid_mechanics/viscoelasticity/ZenerElement.cxx``.
        Einf = self._get_param("Einf", nl_params, Scalar)
        EM = self._get_param("EM", nl_params, Scalar)
        etaM = self._get_param("etaM", nl_params, Scalar)
        E = strain
        Ev = viscous_strain
        # Maxwell branch stress: SM = EM * (E - Ev)
        Ediff = E - Ev
        SM = EM * Ediff
        S = Einf * E + SM
        Ev_dot = SM / etaM
        if v is None:
            return S, Ev_dot

        # Differential pushforwards. The forward is linear in every
        # argument, so every action just scales the incoming tangent by the
        # appropriate (Scalar or SR2) coefficient -- no Jacobian is materialised.
        #   d S      / d E    = (Einf + EM) * I_SR2 -> (Einf + EM) * V
        #   d S      / d Ev   = -EM * I_SR2         -> -EM * V
        #   d Ev_dot / d E    = (EM / etaM) * I_SR2 -> (EM / etaM) * V
        #   d Ev_dot / d Ev   = -(EM / etaM) * I_SR2 -> -(EM / etaM) * V
        #   d S      / d Einf = E                   -> E * V         (V Scalar)
        #   d S      / d EM   = Ediff               -> Ediff * V     (V Scalar)
        #   d Ev_dot / d EM   = Ediff / etaM        -> (Ediff/etaM) * V
        #   d Ev_dot / d etaM = -SM / etaM^2        -> -(SM/etaM^2) * V
        EM_over_etaM = EM / etaM
        Einf_plus_EM = Einf + EM
        actions_S: dict = {
            "strain": lambda V, c=Einf_plus_EM: c * V,
            "viscous_strain": lambda V, c=EM: -(c * V),
        }
        actions_Ev_dot: dict = {
            "strain": lambda V, c=EM_over_etaM: c * V,
            "viscous_strain": lambda V, c=EM_over_etaM: -(c * V),
        }
        if "Einf" in self._nl_params:
            actions_S[self._nl_params["Einf"].input_name] = lambda V, c=E: c * V
        if "EM" in self._nl_params:
            ext = self._nl_params["EM"].input_name
            actions_S[ext] = lambda V, c=Ediff: c * V
            actions_Ev_dot[ext] = lambda V, c=Ediff, e=etaM: (c / e) * V
        if "etaM" in self._nl_params:
            ext = self._nl_params["etaM"].input_name
            actions_Ev_dot[ext] = lambda V, c=SM, e=etaM: -((c / (e * e)) * V)

        # "stress" and "strain" / "viscous_strain" canonical keys translate
        # through ``_var_renames`` inside ``apply_chain_rule`` automatically;
        # the derived ``viscous_strain_rate`` output has no rename entry, so
        # pass the resolved external name (``self._Ev_dot``) directly.
        v_S = self.apply_chain_rule(v, "stress", actions_S, output=S)
        v_Ev_dot = self.apply_chain_rule(v, self._Ev_dot, actions_Ev_dot, output=Ev_dot)
        return S, Ev_dot, {**v_S, **v_Ev_dot}
