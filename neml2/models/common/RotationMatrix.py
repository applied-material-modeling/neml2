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

"""Python-native mirror of C++ ``common/RotationMatrix.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, output
from ...types import (
    R2,
    Rot,
    euler_rodrigues,
    jvp_euler_rodrigues,
)


@register_neml2_object("RotationMatrix")
class RotationMatrix(Model):
    """``orientation_matrix = euler_rodrigues(orientation)``.

    Mirrors C++ ``RotationMatrix`` in ``src/neml2/models/common/RotationMatrix.cxx``.
    The chain-rule action is the closed-form ``R(r) @ skew(omega_b)`` pushforward
    inside :func:`jvp_euler_rodrigues` -- no ``(..., 3, 3, 3)`` derivative
    kernel is ever formed (see the function docstring for the MRP
    body-frame-rate identity).
    """

    hit = HitSchema(
        input("from", Rot, "Rot to convert"),
        output("to", R2, "R2 to store the resulting rotation matrix"),
    )

    def forward(  # type: ignore[override]
        self,
        r: Rot,
        v: ChainRuleDict | None = None,
    ):
        out = euler_rodrigues(r)
        if v is None:
            return out

        # Differential pushforward: dR = ∂(euler_rodrigues)/∂r · dr via
        # the typed JVP primitive (which hides the irreducible 3×3×3 contraction).
        def action(V: Rot) -> R2:
            return jvp_euler_rodrigues(r, V)

        return out, self.apply_chain_rule(v, "to", {"from": action}, output=out)


__all__ = ["RotationMatrix"]
