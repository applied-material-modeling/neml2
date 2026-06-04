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

"""Python-native mirror of the C++ ``VoceIsotropicHardening`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, exp


@register_neml2_object("VoceIsotropicHardening")
class VoceIsotropicHardening(Model):
    r"""Voce isotropic hardening model, $h = R \left[ 1 - \exp(-d \bar{\varepsilon}_p) \right]$,
    where $R$ is the isotropic
    hardening upon saturation, and $d$ is the hardening rate.
    """

    hit = HitSchema(
        input("equivalent_plastic_strain", Scalar, "Equivalent plastic strain"),
        output("isotropic_hardening", Scalar, "Isotropic hardening"),
        parameter(
            "saturated_hardening",
            Scalar,
            "Saturated isotropic hardening",
            attr="R",
            allow_nonlinear=True,
        ),
        parameter(
            "saturation_rate",
            Scalar,
            "Hardening saturation rate",
            attr="d",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``saturated_hardening`` / ``saturation_rate``
    # parameters (stored as ``R`` / ``d``) — no __init__ needed.
    R: Scalar
    d: Scalar

    def forward(  # type: ignore[override]
        self,
        equivalent_plastic_strain: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``VoceIsotropicHardening::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/VoceIsotropicHardening.cxx``.
        eps = equivalent_plastic_strain
        R = self._get_param("R", nl_params, Scalar)
        d = self._get_param("d", nl_params, Scalar)

        # h = R * (1 - exp(-d * eps))
        e = exp(-d * eps)
        h = R * (-e + 1.0)
        if v is None:
            return h

        # Differential pushforward. Each action takes a typed Scalar
        # tangent of the input's type and returns the Scalar tangent of h. No
        # Jacobian is materialised; each coefficient is a Scalar that broadcasts
        # against the leading-K tangent axis automatically.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   dh / d eps = R * d * exp(-d * eps)
        #   dh / d R   = 1 - exp(-d * eps)
        #   dh / d d   = eps * R * exp(-d * eps)
        coef_eps = R * d * e
        actions = {"equivalent_plastic_strain": lambda V, c=coef_eps: c * V}
        if "R" in self._nl_params:
            actions[self._nl_params["R"].input_name] = lambda V, c=(-e + 1.0): c * V
        if "d" in self._nl_params:
            coef_d = eps * R * e
            actions[self._nl_params["d"].input_name] = lambda V, c=coef_d: c * V

        return h, self.apply_chain_rule(v, "isotropic_hardening", actions, output=h)
