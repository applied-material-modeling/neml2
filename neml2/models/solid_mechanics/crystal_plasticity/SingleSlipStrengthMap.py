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

"""Python-native mirror of the C++ ``SingleSlipStrengthMap`` crystal-plasticity leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....factory import register_neml2_object
from ....schema import HitSchema, dependency, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model

if TYPE_CHECKING:
    from ....data import CrystalGeometry


@register_neml2_object("SingleSlipStrengthMap")
class SingleSlipStrengthMap(Model):
    r"""Calculates the slip system strength for all slip systems as
    $\hat{\tau}_i = \bar{\tau} + \tau_0$ where $\hat{\tau}_i$ is
    the strength for slip system i, $\bar{\tau}$ is an evolving slip
    system strength (one value of all systems), defined by another object, and
    $\tau_0$ is a constant strength.
    """

    hit = HitSchema(
        input("slip_hardening", Scalar, "The name of the evolving, scalar strength"),
        output("slip_strengths", Scalar, "Name of the slip system strengths"),
        parameter(
            "constant_strength", Scalar, "The constant slip system strength", allow_nonlinear=True
        ),
        dependency(
            "crystal_geometry",
            "get_data",
            "The name of the Data object containing the crystallographic information "
            "for the material",
            attr="_cg",
            default="crystal_geometry",
        ),
    )
    # Per-slip output from per-crystal input -- the leaf INTRODUCES a
    # "slip" labelled axis on the output (each grain's hardening fan-out
    # to every slip system within that grain).

    # ``from_hit`` auto-declares ``constant_strength`` and stores the resolved
    # ``crystal_geometry`` Data object as ``self._cg`` — no __init__ needed.
    constant_strength: Scalar
    _cg: CrystalGeometry

    def forward(  # type: ignore[override]
        self,
        tau_bar: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        tau_const = self._get_param("constant_strength", nl_params, Scalar)
        # Broadcast to per-slip: insert a length-nslip sub-batch axis at the end.
        nslip = self._cg.nslip
        out = (tau_bar + tau_const).sub_batch.expand_at(nslip, -1)
        if v is None:
            return out

        # ∂τ̂_k/∂τ_bar = 1 for every k; broadcast the per-crystal tangent
        # along a new trailing sub-batch axis of size nslip.
        def tau_bar_action(V: Scalar) -> Scalar:
            return V.sub_batch.expand_at(nslip, -1)

        return out, self.apply_chain_rule(
            v, "slip_strengths", {"slip_hardening": tau_bar_action}, output=out
        )
