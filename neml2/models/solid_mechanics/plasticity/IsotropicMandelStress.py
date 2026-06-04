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

"""Python-native mirror of the C++ ``IsotropicMandelStress`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2


@register_neml2_object("IsotropicMandelStress")
class IsotropicMandelStress(Model):
    """``mandel_stress = cauchy_stress`` (isotropic, small-strain).

    Mirrors C++ ``IsotropicMandelStress``: under isotropic + small-deformation
    assumptions the Mandel stress equals the Cauchy stress in Mandel packing.
    """

    hit = HitSchema(
        input("cauchy_stress", SR2, "Cauchy stress"),
        output("mandel_stress", SR2, "Mandel stress"),
    )

    def forward(  # type: ignore[override]
        self,
        cauchy_stress: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        out = cauchy_stress
        if v is None:
            return out
        return out, self.apply_chain_rule(
            v, "mandel_stress", {"cauchy_stress": lambda V: V}, output=out
        )
