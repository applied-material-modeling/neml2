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

"""Python-native mirror of C++ ``common/Determinant.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import (
    R2,
    SR2,
    Scalar,
    TensorWrapper,
    det,
    inner,
    inv,
)
from ..chain_rule import ChainRuleDict
from ..model import Model


class _Determinant(Model):
    """``J = det(F)`` for a second-order tensor.

    Mirrors the C++ ``Determinant<T>`` (``include/neml2/models/common/Determinant.h``).
    The forward is the determinant; the pushforward is Jacobi's formula
    $∂det(F)/∂F = det(F) · F⁻ᵀ$ collapsed to its directional action
    ``dJ = inner(det(F) · F⁻ᵀ, dF)`` — typed Frobenius contraction, no
    Jacobian materialised.
    """

    _value_type: type[TensorWrapper]


@register_neml2_object("R2Determinant")
class R2Determinant(_Determinant):
    """Determinant of a full ``R2`` tensor."""

    _value_type = R2

    hit = HitSchema(
        input("input", R2, "The second order tensor to calculate the determinant of"),
        output("determinant", Scalar, "The determinant of the input tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        F: R2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        J = det(F)
        if v is None:
            return J
        # ∂det(F)/∂F = det(F) · F⁻ᵀ. The directional pushforward is the
        # Frobenius contraction of that cofactor tensor with ``dF`` — purely
        # typed wrapper algebra, no Jacobian assembled.
        cof = inv(F).base.transpose(-2, -1) * J  # R2 cofactor-transpose, scaled by J

        def F_action(V: R2) -> Scalar:
            return inner(cof, V)

        return J, self.apply_chain_rule(v, "determinant", {"input": F_action}, output=J)


@register_neml2_object("SR2Determinant")
class SR2Determinant(_Determinant):
    """Determinant of a symmetric ``SR2`` tensor."""

    _value_type = SR2

    hit = HitSchema(
        input("input", SR2, "The second order tensor to calculate the determinant of"),
        output("determinant", Scalar, "The determinant of the input tensor"),
    )

    def forward(  # type: ignore[override]
        self,
        F: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        J = det(F)
        if v is None:
            return J
        # Symmetric input ⇒ inv(F) is symmetric ⇒ no transpose. The Frobenius
        # inner of SR2s naturally agrees with the full-tensor product because
        # Mandel packing absorbs the off-diagonal sqrt(2) weights.
        cof = inv(F) * J  # SR2, det(F) · F⁻¹

        def F_action(V: SR2) -> Scalar:
            return inner(cof, V)

        return J, self.apply_chain_rule(v, "determinant", {"input": F_action}, output=J)


__all__ = ["R2Determinant", "SR2Determinant"]
