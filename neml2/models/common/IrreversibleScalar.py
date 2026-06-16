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

"""Python-native mirror of the C++ ``IrreversibleScalar`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, derived_input, input, output
from ...types import Scalar, gt, where
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("IrreversibleScalar")
class IrreversibleScalar(Model):
    r"""Monotonically-increasing ratchet on a Scalar:
    $y = \max(y_{n-1}, x)$ where $x$ is the trial value (`from`)
    and $y_{n-1}$ is the previous-step value of the output
    (auto-declared via `history_name`)."""

    hit = HitSchema(
        input("from", Scalar, "Trial value at the current step"),
        output("to", Scalar, "Updated (irreversibly-capped) value"),
        # Previous-step value of the output is auto-declared by appending the
        # ``~1`` history suffix to the resolved output name (matches the C++
        # ``history_name(_to.name(), /*nstep=*/1)`` plumbing). No HIT knob.
        derived_input("to", Scalar, attr="_to_old", suffix="~1"),
    )

    # ``_to_old`` carries the resolved ``<to>~1`` history input name; the
    # canonical ``from`` / ``to`` names are translated by ``apply_chain_rule``.
    _to_old: str

    def forward(  # type: ignore[override]
        self,
        x: Scalar,
        x_old: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Forward: y = where(x > x_old, x, x_old) = max(x_old, x). The mask is
        # a bool tensor (no grad), matching the C++ ``advance_mask.detach()``.
        advance = gt(x, x_old)
        y = where(advance, x, x_old)
        if v is None:
            return y

        # D-062 pushforward: the Jacobian collapses to 1 on the advancing
        # branch and 0 on the frozen branch, with the complementary partial
        # routed to the history input. Each input's action is the masked
        # tangent on its branch and a structural zero on the other.
        zero = Scalar.from_value(0.0, like=x)

        def x_action(V: Scalar, mask: Scalar = advance, z: Scalar = zero) -> Scalar:
            return where(mask, V, z)

        def x_old_action(V: Scalar, mask: Scalar = advance, z: Scalar = zero) -> Scalar:
            return where(mask, z, V)

        return y, self.apply_chain_rule(
            v,
            "to",
            {"from": x_action, self._to_old: x_old_action},
            output=y,
        )


__all__ = ["IrreversibleScalar"]
