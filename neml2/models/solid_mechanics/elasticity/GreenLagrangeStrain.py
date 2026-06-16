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

"""Python-native mirror of the C++ ``GreenLagrangeStrain`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import R2, SR2, sym
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("GreenLagrangeStrain")
class GreenLagrangeStrain(Model):
    r"""Green-Lagrange strain, $E = \frac{1}{2} (C - I)$, where
    $C = F^T F$ is the right Cauchy-Green tensor and $I$ is the
    identity tensor."""

    hit = HitSchema(
        input("deformation_gradient", R2, "The deformation gradient"),
        output("strain", SR2, "The Green-Lagrange strain"),
    )

    def forward(  # type: ignore[override]
        self,
        deformation_gradient: R2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        F = deformation_gradient
        # C = F^T F is symmetric by construction; sym() packs (C + C^T)/2 into
        # Mandel storage, matching the C++ ``SR2(C)`` conversion.
        C = F.base.transpose(-2, -1) @ F
        I = SR2.identity(dtype=F.dtype, device=F.device)
        E = 0.5 * (sym(C) - I)
        if v is None:
            return E

        # Differential pushforward:
        #   dE = 1/2 (dF^T F + F^T dF) = sym(F^T dF)
        # since dF^T F = (F^T dF)^T, and sym(A) packs (A + A^T)/2 to Mandel.
        # Pure typed wrapper algebra; no R3 derivative kernel built.
        def F_action(V: R2) -> SR2:
            return sym(F.base.transpose(-2, -1) @ V)

        return E, self.apply_chain_rule(v, "strain", {"deformation_gradient": F_action}, output=E)
