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

from ....factory import register_neml2_object
from ....schema import HitSchema, derived_input, input, output, parameter
from ....types import SR2, Scalar, lt, norm, where
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("CrystalPlasticityStrainPredictor")
class CrystalPlasticityStrainPredictor(Model):
    r"""Warm-up predictor for crystal plasticity models. Computes an initial guess
    for the elastic strain as $\varepsilon^e = s \cdot \Delta t \cdot d$
    where $\Delta t = t - t_n$ is the time increment, $d$ is the
    deformation rate, and $s$ is a scale factor.
    """

    hit = HitSchema(
        input("deformation_rate", SR2, "Deformation rate", attr="_D"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        output("elastic_strain", SR2, "Predicted elastic strain", attr="_Ee"),
        derived_input("elastic_strain", SR2, attr="_Ee_n", suffix="~1"),
        parameter("scale", Scalar, "Scale factor applied to the strain increment", default="1.0"),
        parameter(
            "threshold",
            Scalar,
            "Only apply the predictor if the old elastic strain norm is below this threshold.",
            default="1e-3",
        ),
    )

    _D: str
    _t: str
    _t_n: str
    _Ee: str
    _Ee_n: str
    scale: Scalar
    threshold: Scalar

    def forward(  # type: ignore[override]
        self,
        D: SR2,
        t: Scalar,
        t_n: Scalar,
        Ee_n: SR2,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        dt = t - t_n
        scale = self._get_param("scale", promoted_params, Scalar)
        threshold = self._get_param("threshold", promoted_params, Scalar)
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
