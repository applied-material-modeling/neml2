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

"""Python-native mirror of C++ ``solid_mechanics/kinematics/VolumeAdjustDeformationGradient.h``."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import R2, Scalar, pow


@register_neml2_object("VolumeAdjustDeformationGradient")
class VolumeAdjustDeformationGradient(Model):
    r"""Calculate the volume-adjusted deformation gradient, i.e. $F_e = J^{-\frac{1}/{3}} F$, where
    $F$ is the pre-adjusted deformation
    gradient and $J$ is the total jacobian of the volumetric deformation
    gradients to be removed.
    """

    hit = HitSchema(
        input("input", R2, "Input deformation gradient"),
        input(
            "jacobian",
            Scalar,
            "The Jacobian that controls the volume adjustment of the input deformation gradient",
        ),
        output("output", R2, "Output adjusted deformation gradient"),
    )

    def forward(  # type: ignore[override]
        self,
        F: R2,
        J: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Forward: Fe = F * J^(-1/3) -- typed wrapper algebra
        # (pow(Scalar, float) -> Scalar; R2 * Scalar -> R2). No raw torch on .data.
        s: Scalar = pow(J, -1.0 / 3.0)  # type: ignore[assignment]
        Fe = F * s
        if v is None:
            return Fe

        # D-062 pushforward: differentiate Fe = F * J^(-1/3) by the product rule.
        #   dFe/dF[dF] = J^(-1/3) * dF
        #   dFe/dJ[dJ] = -(1/3) * J^(-4/3) * F * dJ
        # Both branches stay in typed R2 * Scalar / R2 * float wrapper algebra,
        # mirroring the C++ ``_Fe.d(_F) = J^(-1/3) * I`` and
        # ``_Fe.d(_J) = -1/3 * J^(-4/3) * F`` derivatives.
        dFe_dJ_coeff: Scalar = pow(J, -4.0 / 3.0) * (-1.0 / 3.0)  # type: ignore[assignment]
        F_coeff: R2 = F * dFe_dJ_coeff

        def F_action(dF: R2, s_: Scalar = s) -> R2:
            return dF * s_

        def J_action(dJ: Scalar, coeff: R2 = F_coeff) -> R2:
            return coeff * dJ

        return Fe, self.apply_chain_rule(
            v,
            "output",
            {"input": F_action, "jacobian": J_action},
            output=Fe,
        )


__all__ = ["VolumeAdjustDeformationGradient"]
