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

"""Python-native mirror of the C++ ``ResolvedShear`` crystal-plasticity leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, dependency, input, output
from ....types import R2, SR2, Scalar, inner, jvp_rotate_sym, rotate_sym

if TYPE_CHECKING:
    from ....data import CrystalGeometry


@register_native("ResolvedShear")
class ResolvedShear(Model):
    r"""Calculates the resolved shears as
    $\tau_i = \sigma : Q \operatorname{sym}\left(d_i \otimes n_i \right) Q^T$ where $\tau_i$ is the
    resolved shear on slip system i,
    $\sigma$ is the Cauchy stress $Q$ is the orientation matrix,
    $d_i$ is the slip direction, and $n_i$ is the slip system
    normal.
    """

    hit = HitSchema(
        input("stress", SR2, "The name of the Cauchy stress tensor"),
        input("orientation_matrix", R2, "The name of the orientation matrix"),
        output("resolved_shears", Scalar, "The name of the resolved shears"),
        dependency(
            "crystal_geometry",
            "get_data",
            "The name of the data object with the crystallographic information",
            default="crystal_geometry",
        ),
    )
    # Rotating per-crystal R into per-slip Schmid combines per-crystal data with
    # per-slip data; the output has sub_batch_ndim=1 (per-slip). Each per-slip
    # output entry depends on one σ and one R — for chain rule purposes this is
    # "dense" because the sub-batch axis is added at this leaf.
    list_deriv = {
        ("resolved_shears", "stress"): "dense",
        ("resolved_shears", "orientation_matrix"): "dense",
    }

    def __init__(self, *, crystal_geometry: CrystalGeometry) -> None:
        super().__init__()
        self._cg = crystal_geometry

    def forward(  # type: ignore[override]
        self,
        stress: SR2,
        R: R2,
        v: ChainRuleDict | None = None,
    ):
        # Per-slip Schmid M (sub_batch_ndim=1, shape (..., nslip, 6)) rotated by
        # per-crystal R. D-044 means we can't trust input wrappers'
        # ``sub_batch_ndim`` (resets to 0 across ComposedModel); operate on raw
        # tensors so PyTorch's right-aligned broadcast lines up nslip cleanly.
        # ``M.data``'s nslip is at axis -2; ``stress.data.unsqueeze(-2)`` puts
        # a singleton at axis -2 so the multiply broadcasts (1 vs nslip → nslip).
        M = self._cg.M  # SR2 sub_batch=1
        R_sb = R.sub_batch_unsqueeze(-1)
        M_rot = rotate_sym(M, R_sb)  # SR2 (..., nslip, 6)
        stress_sb = stress.sub_batch_unsqueeze(-1)
        rss = inner(M_rot, stress_sb)
        if v is None:
            return rss

        # Differential pushforwards, pure typed-wrapper algebra:
        #   stress: dτ = M_rot : dσ
        #   R:      dτ = d(sym(R M Rᵀ)) : σ via the jvp_rotate_sym primitive
        #           (product rule in 3×3, no leaf-level Jacobian).
        def stress_action(V: SR2) -> Scalar:
            return inner(M_rot, V.sub_batch_unsqueeze(-1))

        def R_action(V: R2) -> Scalar:
            return inner(jvp_rotate_sym(M, R_sb, V.sub_batch_unsqueeze(-1)), stress_sb)

        return rss, self.apply_chain_rule(
            v,
            "resolved_shears",
            {"stress": stress_action, "orientation_matrix": R_action},
            output=rss,
        )
