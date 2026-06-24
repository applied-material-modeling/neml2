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

"""Python-native mirror of the C++ ``CrystalPlasticityStrainPredictor`` leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import HitSchema, parameter
from ....types import SR2, Scalar, lt, norm, where
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    from ....factory import _NativeInputFile


@register_neml2_object("CrystalPlasticityStrainPredictor")
class CrystalPlasticityStrainPredictor(Model):
    r"""Warm-up predictor for crystal plasticity models. Computes an initial guess
    for the elastic strain as $\varepsilon^e = s \cdot \Delta t \cdot d$
    where $\Delta t = t - t_n$ is the time increment, $d$ is the
    deformation rate, and $s$ is a scale factor.
    """

    hit = HitSchema(
        parameter("scale", Scalar, "Scale factor applied to the strain increment", default="1.0"),
        parameter(
            "threshold",
            Scalar,
            "Only apply the predictor if the old elastic strain norm is below this threshold.",
            default="1e-3",
        ),
    )

    scale: Scalar
    threshold: Scalar

    def __init__(
        self,
        scale=1.0,
        threshold=1e-3,
        *,
        deformation_rate: str = "deformation_rate",
        time: str = "t",
        elastic_strain: str = "elastic_strain",
        factory: _NativeInputFile | None = None,
    ) -> None:
        super().__init__()
        self._D = deformation_rate
        self._t = time
        self._t_n = f"{time}~1"
        self._Ee = elastic_strain
        self._Ee_n = f"{elastic_strain}~1"
        self.input_spec = {
            deformation_rate: SR2,
            self._t: Scalar,
            self._t_n: Scalar,
            self._Ee_n: SR2,
        }
        self.output_spec = {elastic_strain: SR2}
        self.declare_typed_parameter("scale", scale, Scalar, factory=factory)
        self.declare_typed_parameter("threshold", threshold, Scalar, factory=factory)

    def forward(  # type: ignore[override]
        self,
        D: SR2,
        t: Scalar,
        t_n: Scalar,
        Ee_n: SR2,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        dt = t - t_n
        scale = self._get_param("scale", nl_params, Scalar)
        threshold = self._get_param("threshold", nl_params, Scalar)
        Ee_pred = Ee_n + scale * D * dt
        use_pred = lt(norm(Ee_n), threshold)
        result = where(use_pred, Ee_pred, Ee_n)
        if v is None:
            return result

        # Predictors are not differentiated in the Newton residual (they're
        # one-shot warm-ups). Provide trivial pass-throughs so chain-rule
        # callers don't error — zero tangents that preserve type/metadata.
        def trivial(V: SR2) -> SR2:
            return V * 0.0

        return result, self.apply_chain_rule(
            v,
            self._Ee,
            {
                self._D: trivial,
                self._t: trivial,
                self._t_n: trivial,
                self._Ee_n: trivial,
            },
            output=result,
        )
