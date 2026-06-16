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

"""Python-native mirror of the C++ ``PlasticDeformationRate`` crystal-plasticity leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import HitSchema, dependency, input, output
from ....types import R2, SR2, Scalar, jvp_rotate, rotate, sum
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    from ....data import CrystalGeometry


@register_neml2_object("PlasticDeformationRate")
class PlasticDeformationRate(Model):
    r"""Caclulates the plastic deformation rate as
    $d^p = \sum_{i=1}^{n_{slip}} \dot{\gamma}_i Q \operatorname{sym}{\left(d_i \otimes n_i \right)} Q^T$
    with
    $d^p$ the plastic deformation rate, $\dot{\gamma}_i$ the slip
    rate on the ith slip system, $Q$ the orientation, $d_i$ the slip
    system direction, and $n_i$ is the slip system normal.
    """  # noqa: E501

    hit = HitSchema(
        input("orientation_matrix", R2, "The name of the orientation matrix"),
        input("slip_rates", Scalar, "The name of the tensor containg the current slip rates"),
        output("plastic_deformation_rate", SR2, "The name of the plastic deformation rate tensor"),
        dependency(
            "crystal_geometry",
            "get_data",
            "The name of the Data object containing the crystallographic information "
            "for the material",
            default="crystal_geometry",
        ),
    )

    def __init__(self, *, crystal_geometry: CrystalGeometry) -> None:
        super().__init__()
        self._cg = crystal_geometry

    def forward(  # type: ignore[override]
        self,
        R: R2,
        g: Scalar,
        v: ChainRuleDict | None = None,
    ):
        M = self._cg.M  # SR2 sub_batch_ndim=1, shape (nslip, 6)
        weighted = M * g
        dp_cry = sum(weighted.sub_batch, -1)
        # Rotate to lab frame: dp_lab = sym(R · dp_cry_full · R^T)
        dp = rotate(dp_cry, R)
        if v is None:
            return dp

        # Differential pushforwards:
        #   R:  d(dp) = jvp_rotate(dp_cry, R, dR)  (product rule, typed)
        #   γ̇: dp = rotate(·, R) is LINEAR in dp_cry = Σ γ̇_k M_k, so push the
        #       slip-rate tangent through the same sum + rotation, no Jacobian.
        def R_action(V: R2) -> SR2:
            return jvp_rotate(dp_cry, R, V)

        def g_action(V: Scalar) -> SR2:
            return rotate(sum((M * V).sub_batch, -1), R)

        return dp, self.apply_chain_rule(
            v,
            "plastic_deformation_rate",
            {"orientation_matrix": R_action, "slip_rates": g_action},
            output=dp,
        )
