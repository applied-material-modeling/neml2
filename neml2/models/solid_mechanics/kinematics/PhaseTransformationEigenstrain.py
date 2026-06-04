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

"""Python-native mirror of C++ ``solid_mechanics/kinematics/PhaseTransformationEigenstrain.h``."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar


@register_neml2_object("PhaseTransformationEigenstrain")
class PhaseTransformationEigenstrain(Model):
    r"""Define the (cummulative, as opposed to instantaneous) linear isotropic phase
    transformation (from phase A to phase B) eigenstrain, i.e.
    $\boldsymbol{\varepsilon}_\mathrm{PT} = \Delta V f \boldsymbol{I}$, where $\Delta V$ is the
    volume fraction change when going from phase A to B, $f$
    is the phase fraction (0 to 1, A to B).
    """

    # Forward: ``eg = f * dv * I`` where ``I`` is the SR2 identity. The leaf is
    # bilinear in ``(f, dv)`` -- linear in each separately holding the other
    # fixed. Pure typed wrapper algebra: ``SR2.identity`` and the
    # ``Scalar * Scalar * SR2`` operators handle Mandel packing and sub-batch
    # alignment, so no ``torch.<op>(.data)`` calls appear in either the forward
    # or the chain-rule action bodies.

    hit = HitSchema(
        input("phase_fraction", Scalar, "Phase fraction"),
        input(
            "volume_fraction_change",
            Scalar,
            "Change in volume fraction going from phase A to phase B",
        ),
        output("eigenstrain", SR2, "Eigenstrain"),
    )

    def forward(  # type: ignore[override]
        self,
        phase_fraction: Scalar,
        volume_fraction_change: Scalar,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        f = phase_fraction
        dv = volume_fraction_change
        I = SR2.identity(dtype=f.data.dtype, device=f.data.device)
        eg = f * dv * I
        if v is None:
            return eg

        # Differential pushforward: d(eg)/df = dv * I and
        # d(eg)/d(dv) = f * I, so each action scales the incoming Scalar tangent
        # by the held coefficient and multiplies by the identity SR2. No SSR4
        # ever constructed, no Jacobian materialised.
        def phase_fraction_action(V: Scalar) -> SR2:
            return dv * V * I

        def volume_fraction_change_action(V: Scalar) -> SR2:
            return f * V * I

        return eg, self.apply_chain_rule(
            v,
            "eigenstrain",
            {
                "phase_fraction": phase_fraction_action,
                "volume_fraction_change": volume_fraction_change_action,
            },
            output=eg,
        )


__all__ = ["PhaseTransformationEigenstrain"]
