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

"""Python-native mirror of C++ ``solid_mechanics/kinematics/VolumeChangeEigenstrain.h``."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import SR2, Scalar, pow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("VolumeChangeEigenstrain")
class VolumeChangeEigenstrain(Model):
    r"""Define the (cumulative, as opposed to instantaneous) linear isotropic
    volume expansion eigenstrain, i.e.
    $\boldsymbol{\varepsilon}_V = (\frac{V}{V0})^(1/3)-1 \boldsymbol{I}$, where $V$ is the current
    volume, and $V0$ is the reference (initial) volume.
    """

    hit = HitSchema(
        input("volume", Scalar, "Volume"),
        output("eigenstrain", SR2, "Eigenstrain"),
        parameter("reference_volume", Scalar, "Reference (initial) volume", attr="V0"),
    )

    # ``from_hit`` auto-declares the ``reference_volume`` parameter (stored as
    # ``V0``). Annotate so pyright sees the typed wrapper that
    # ``Model.__getattr__`` returns.
    V0: Scalar

    def forward(  # type: ignore[override]
        self,
        volume: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        V0 = self._get_param("V0", promoted_params, Scalar)
        I = SR2.identity(dtype=volume.dtype, device=volume.device)
        # Forward: eg = ((V/V0)^(1/3) - 1) * I -- typed wrapper algebra
        # (Scalar / Scalar -> Scalar; pow(Scalar, float) -> Scalar; Scalar - 1.0
        # -> Scalar; Scalar * SR2 -> SR2). No raw torch on .data.
        q = volume / V0
        scale = pow(q, 1.0 / 3.0) - 1.0
        eg = scale * I
        if v is None:
            return eg

        # D-062 pushforward: deg/dV = (1/(3*V0)) * (V/V0)^(-2/3) * I, so the
        # action along a Scalar tangent dV is that coefficient times dV times I,
        # written entirely in typed wrapper algebra. Mirrors the C++
        # ``_eg.d(_V) = 1/(3*V0) * pow(_V/_V0, -2/3) * SR2::identity(...)``.
        coeff = pow(q, -2.0 / 3.0) / (V0 * 3.0)

        def volume_action(V_t: Scalar, c=coeff, I_=I) -> SR2:
            return (c * V_t) * I_

        return eg, self.apply_chain_rule(v, "eigenstrain", {"volume": volume_action}, output=eg)


__all__ = ["VolumeChangeEigenstrain"]
