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

"""Python-native mirror of the C++ ``MacaulaySplit`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar, heaviside, macaulay
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("MacaulaySplit")
class MacaulaySplit(Model):
    r"""Split a Scalar into its Macaulay (positive) and negative parts:
    $\langle x \rangle_+ = \max(x, 0)$ and
    $\langle x \rangle_- = x - \langle x \rangle_+$."""

    hit = HitSchema(
        input("from", Scalar, "The Scalar to split"),
        output("to_positive", Scalar, "Name of the Macaulay (positive) part output"),
        output("to_negative", Scalar, "Name of the negative-part output"),
    )

    def forward(  # type: ignore[override]
        self,
        x: Scalar,
        v: ChainRuleDict | None = None,
    ):
        # Forward: <x>_+ = macaulay(x); <x>_- = x - <x>_+.
        x_pos = macaulay(x)
        pos = x_pos
        neg = x - x_pos
        if v is None:
            return pos, neg

        # Differential pushforward: d<x>_+/dx = H(x); d<x>_-/dx = 1 - H(x).
        # H is element-wise so the action is just the per-element weight times
        # the incoming Scalar tangent V.
        H = heaviside(x)
        one_minus_H = 1.0 - H

        def pos_action(V: Scalar, c: Scalar = H) -> Scalar:
            return c * V

        def neg_action(V: Scalar, c: Scalar = one_minus_H) -> Scalar:
            return c * V

        v_pos = self.apply_chain_rule(v, "to_positive", {"from": pos_action}, output=pos)
        v_neg = self.apply_chain_rule(v, "to_negative", {"from": neg_action}, output=neg)
        # Merge dicts (different outer keys).
        return pos, neg, {**v_pos, **v_neg}


__all__ = ["MacaulaySplit"]
