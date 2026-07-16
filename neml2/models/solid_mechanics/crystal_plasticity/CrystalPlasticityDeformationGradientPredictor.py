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

"""Fp warm-up predictor for multiplicative crystal plasticity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import HitSchema, parameter
from ....types import R2, SR2, Scalar, inv, lt, norm, sym, where
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    from ....factory import _NativeInputFile


@register_neml2_object("CrystalPlasticityDeformationGradientPredictor")
class CrystalPlasticityDeformationGradientPredictor(Model):
    r"""Warm-up predictor for multiplicative crystal plasticity.

    Seeds the plastic deformation gradient so the local Newton starts from a
    low-stress state instead of the frozen-$F_p$ elastic trial (which over-stresses
    for a large step and inflates the initial residual). With the trial elastic
    deformation $F_e^{tr} = F F_{p,n}^{-1}$, the guess is

    $$ F_e^{pred} = I + s (F_e^{tr} - I), \qquad F_p^{pred} = (F_e^{pred})^{-1} F, $$

    where the scale $s < 1$ relaxes the over-shot trial elastic strain back toward
    the yield surface and the remainder is absorbed into $F_p$. The guess is applied
    only where the trial elastic strain norm exceeds ``threshold`` (i.e. the frozen
    trial over-stresses); elsewhere $F_{p,n}$ is kept. This mirrors
    :class:`CrystalPlasticityStrainPredictor` for the $F_p$-based (multiplicative)
    formulation. It is a one-shot warm-up: its derivative does not enter the Newton
    residual, so the chain rule is a trivial pass-through.
    """

    hit = HitSchema(
        parameter(
            "scale",
            Scalar,
            "Scale factor applied to the trial elastic deformation increment",
            default="0.1",
        ),
        parameter(
            "threshold",
            Scalar,
            "Apply the predictor only where the trial elastic strain norm exceeds this value.",
            default="1e-3",
        ),
    )

    scale: Scalar
    threshold: Scalar

    def __init__(
        self,
        scale=0.1,
        threshold=1e-3,
        *,
        deformation_gradient: str = "deformation_gradient",
        plastic_deformation_gradient: str = "plastic_deformation_gradient",
        factory: _NativeInputFile | None = None,
    ) -> None:
        super().__init__()
        self._F = deformation_gradient
        self._Fp = plastic_deformation_gradient
        self._Fp_n = f"{plastic_deformation_gradient}~1"
        self.input_spec = {
            deformation_gradient: R2,
            self._Fp_n: R2,
        }
        self.output_spec = {plastic_deformation_gradient: R2}
        self.declare_typed_parameter("scale", scale, Scalar, factory=factory)
        self.declare_typed_parameter("threshold", threshold, Scalar, factory=factory)

    def forward(  # type: ignore[override]
        self,
        F: R2,
        Fp_n: R2,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        scale = self._get_param("scale", promoted_params, Scalar)
        threshold = self._get_param("threshold", promoted_params, Scalar)
        I_R2 = R2.identity(dtype=F.dtype, device=F.device)
        I_SR2 = SR2.identity(dtype=F.dtype, device=F.device)

        Fe_trial = F @ inv(Fp_n)  # frozen-Fp elastic trial (over-stresses)
        # Linearized trial elastic strain sym(Fe) - I -- adequate for the gate.
        Ee_trial = sym(Fe_trial) - I_SR2
        # Relax the over-shot elastic trial toward yield; absorb the rest into Fp.
        Fe_pred = I_R2 + scale * (Fe_trial - I_R2)
        Fp_pred = inv(Fe_pred) @ F
        # Apply only where the frozen trial over-stresses (trial strain > threshold).
        use_pred = lt(threshold, norm(Ee_trial))
        result = where(use_pred, Fp_pred, Fp_n)
        if v is None:
            return result

        # One-shot warm-up: not differentiated in the Newton residual. Trivial
        # zero tangents keep chain-rule callers from erroring (mirrors
        # CrystalPlasticityStrainPredictor / ConstantExtrapolationPredictor).
        def trivial(V: R2) -> R2:
            return V * 0.0

        return result, self.apply_chain_rule(
            v,
            self._Fp,
            {self._F: trivial, self._Fp_n: trivial},
            output=result,
        )
