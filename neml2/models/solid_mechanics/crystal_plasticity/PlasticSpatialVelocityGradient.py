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

"""Python-native mirror of the C++ ``PlasticSpatialVelocityGradient`` crystal-plasticity leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, dependency, input, output
from ....types import R2, Scalar, jvp_rotate, rotate, sum

if TYPE_CHECKING:
    from ....data import CrystalGeometry


@register_native("PlasticSpatialVelocityGradient")
class PlasticSpatialVelocityGradient(Model):
    r"""Caclulates the plastic spatial velocity gradient as
    $l^p = \sum_{i=1}^{n_{slip}} \dot{\gamma}_i Q \left(d_i \otimes n_i \right) Q^T$
    with $l^p$ the plastic spatial velocity gradient, $\dot{\gamma}_i$
    the slip rate on the ith slip system, $Q$ the orientation, $d_i$
    the slip system direction, and $n_i$ the slip system normal.
    """

    hit = HitSchema(
        input("orientation_matrix", R2, "The name of the orientation matrix"),
        input("slip_rates", Scalar, "The name of the tensor containg the current slip rates"),
        output(
            "plastic_spatial_velocity_gradient",
            R2,
            "The name of the plastic spatial velocity gradient",
        ),
        dependency(
            "crystal_geometry",
            "get_data",
            "The name of the Data object containing the crystallographic information "
            "for the material",
            default="crystal_geometry",
        ),
    )
    list_deriv = {("plastic_spatial_velocity_gradient", "slip_rates"): "dense"}

    def __init__(self, *, crystal_geometry: CrystalGeometry) -> None:
        super().__init__()
        self._cg = crystal_geometry

    def forward(  # type: ignore[override]
        self,
        R: R2,
        g: Scalar,
        v: ChainRuleDict | None = None,
    ):
        A = self._cg.A  # R2 sub_batch_ndim=1, the full d_i (x) n_i Schmid tensor
        weighted = A * g
        lp_cry = cast(R2, sum(weighted.sub_batch, -1))
        # Rotate to lab frame: l^p = R · lp_cry · R^T (no sym/skew projection).
        lp = rotate(lp_cry, R)
        if v is None:
            return lp

        # Differential pushforwards, mirror of PlasticDeformationRate:
        #   R:  d(lp) = jvp_rotate(lp_cry, R, dR)  (product rule, typed)
        #   gamma_dot: rotate(., R) is LINEAR in lp_cry = Sum_k gamma_dot_k A_k,
        #             so push the slip-rate tangent through the same sum + rotation,
        #             no Jacobian.
        def R_action(V: R2) -> R2:
            return jvp_rotate(lp_cry, R, V)

        def g_action(V: Scalar) -> R2:
            return rotate(cast(R2, sum((A * V).sub_batch, -1)), R)

        return lp, self.apply_chain_rule(
            v,
            "plastic_spatial_velocity_gradient",
            {"orientation_matrix": R_action, "slip_rates": g_action},
            output=lp,
        )
