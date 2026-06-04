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

"""Python-native mirror of the C++ ``GursonCavitation`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar
from ....types.functions import tr


@register_neml2_object("GursonCavitation")
class GursonCavitation(Model):
    r"""Local mass balance used in conjunction with the GTNYieldFunction,
    $\dot{\phi} = (1-\phi) \dot{\varepsilon}_p$.
    """

    hit = HitSchema(
        input("plastic_strain_rate", SR2, "Plastic strain rate"),
        input("void_fraction", Scalar, "Void fraction (porosity)"),
        output("void_fraction_rate", Scalar, "Rate of void evolution"),
    )

    def forward(  # type: ignore[override]
        self,
        plastic_strain_rate: SR2,
        void_fraction: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        Ep_dot = plastic_strain_rate
        phi = void_fraction
        ep_dot = tr(Ep_dot)
        phi_dot = (1 - phi) * ep_dot
        if v is None:
            return phi_dot
        # Differential pushforward:
        #   d(phi_dot)/d(phi) * V_phi      = -tr(Ep_dot) * V_phi  (Scalar)
        #   d(phi_dot)/d(Ep_dot) : V_Ep    = (1 - phi) * tr(V_Ep) (Scalar)
        # Both reduce via the trace primitive -- no SSR4 / R4 ever formed.
        return phi_dot, self.apply_chain_rule(
            v,
            "void_fraction_rate",
            {
                "void_fraction": lambda V, c=ep_dot: -c * V,
                "plastic_strain_rate": lambda V, c=(1 - phi): c * tr(V),
            },
            output=phi_dot,
        )
