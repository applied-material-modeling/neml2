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

"""Python-native mirror of C++ ``common/SR2Invariant.h``."""

from __future__ import annotations

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output
from ...types import (
    SR2,
    Scalar,
    dev,
    inner,
    sqrt,
    tr,
)
from ..chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ..model import Model


@register_neml2_object("SR2Invariant")
class SR2Invariant(Model):
    """Scalar invariants of an ``SR2`` tensor.

    Supported ``invariant_type``: ``VONMISES``, ``EFFECTIVE_STRAIN``, ``I1``.
    ``I2`` is an easy follow-on when a HIT file uses it.
    """

    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input(
            "tensor", SR2, "Symmetric second order tensor to take the invariant of", attr="_tensor"
        ),
        output("invariant", Scalar, "Invariant", attr="_invariant"),
        option(
            "invariant_type",
            str,
            "Type of invariant. Options are: INVALID, EFFECTIVE_STRAIN, VONMISES, I2, I1",
            attr="_type",
        ),
    )

    _tensor: str
    _invariant: str
    _type: str

    def __post_init__(self) -> None:
        if self._type not in ("VONMISES", "EFFECTIVE_STRAIN", "I1"):
            raise NotImplementedError(
                f"SR2Invariant: invariant_type={self._type!r} not yet implemented "
                "(only VONMISES, EFFECTIVE_STRAIN, and I1 are supported)"
            )

    def forward(  # type: ignore[override]
        self,
        tensor: SR2,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        if self._type == "I1":
            return self._forward_i1(tensor, v, v2, vh)
        # VONMISES = sqrt((3/2) S:S), EFFECTIVE_STRAIN = sqrt((2/3) S:S);
        # only the coefficient inside the sqrt (and the chain-rule prefactor)
        # differs. Mirrors C++ ``SR2Invariant::set_value``.
        coeff = 1.5 if self._type == "VONMISES" else 2.0 / 3.0
        return self._forward_deviatoric_norm(tensor, coeff, v, v2, vh)

    def _forward_i1(
        self,
        tensor: SR2,
        v: ChainRuleDict | None,
        v2: SecondOrderChainRuleDict | None,
        vh: ChainRuleDict | None,
    ):
        # I1 = tr(tensor). Linear in tensor, so the differential pushforward is
        # simply ``tr(V)`` and the Hessian vanishes (zero contribution).
        out = tr(tensor)
        if v is None:
            return out

        def action(V: SR2) -> Scalar:
            return tr(V)

        actions_1 = {self._tensor: action}
        if v2 is None and vh is None:
            return out, *self.propagate_tangents(v, self._invariant, actions_1, output=out)

        # Hessian of a linear map is zero; omit ``actions_2`` so
        # ``propagate_tangents`` contributes nothing to the second-order chain.
        return out, *self.propagate_tangents(
            v,
            self._invariant,
            actions_1,
            output=out,
            v2=v2,
            actions_2={},
            vh=vh,
        )

    def _forward_deviatoric_norm(
        self,
        tensor: SR2,
        coeff: float,
        v: ChainRuleDict | None,
        v2: SecondOrderChainRuleDict | None,
        vh: ChainRuleDict | None,
    ):
        # ``r = sqrt(coeff · S:S)`` with ``S = dev(tensor)``. VONMISES has
        # ``coeff = 3/2``; EFFECTIVE_STRAIN has ``coeff = 2/3``. The
        # chain-rule structure is identical — only the coefficient changes.
        S = dev(tensor)
        eps = torch.finfo(tensor.dtype).eps
        out = sqrt(coeff * inner(S, S) + eps)
        if v is None:
            return out
        # Differential pushforward. ∂r/∂tensor = N = coeff · S / r
        # (the flow direction, an SR2). First-order action is the colon
        # product inner(N, V) — pure typed algebra, no Jacobian.
        N = S * (coeff / out)  # SR2

        def action(V: SR2) -> Scalar:
            return inner(N, V)

        actions_1 = {self._tensor: action}
        if v2 is None and vh is None:
            return out, *self.propagate_tangents(v, self._invariant, actions_1, output=out)

        # Second-order Hessian form: ∂²r/∂M² = (1/r)[coeff · P_dev − N ⊗ N].
        # action_2 receives ONE primal-shape
        # tangent per slot (no leading seed dim); the framework's
        # ``_apply_action_2`` handles the (N_a, N_b) seed-pair iteration and
        # restacks. Body is pure typed-wrapper algebra — no .data, no outer.
        inv_r = 1.0 / out

        def action_2(Va: SR2, Vb: SR2) -> Scalar:
            return inv_r * (coeff * inner(Va, dev(Vb)) - inner(N, Va) * inner(N, Vb))

        return out, *self.propagate_tangents(
            v,
            self._invariant,
            actions_1,
            output=out,
            v2=v2,
            actions_2={(self._tensor, self._tensor): action_2},
            vh=vh,
        )


__all__ = ["SR2Invariant"]
