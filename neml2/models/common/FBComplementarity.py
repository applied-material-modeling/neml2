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

"""Python-native mirror of C++ ``common/FBComplementarity.h``."""

from __future__ import annotations

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_neml2_object
from ...model import Model
from ...schema import HitSchema, input, option, output
from ...types import (
    Scalar,
    sqrt,
)


@register_neml2_object("FBComplementarity")
class FBComplementarity(Model):
    r"""If $a \ge 0, b \ge 0, ab = 0$ then the Fischer Burmeister (FB)
    complementarity condition is:$r = a+b-\sqrt(a^2+b^2) = 0$.
    """

    hit = HitSchema(
        input("a", Scalar, "First condition", attr="_a"),
        input("b", Scalar, "Second condition", attr="_b"),
        # The HIT option key matches the C++ ``options.add_output("complementarity",
        # ...)`` so a user wiring this leaf with ``complementarity = 'd_residual'``
        # gets the variable renamed and the host NonlinearSystem's
        # ``<unknown>_residual`` inference works out of the box.
        output(
            "complementarity",
            Scalar,
            "The Fischer Burmeister complementarity condition",
            attr="_residual",
        ),
        option(
            "a_inequality",
            str,
            "Type of the inequality for the first variable a.",
            default="LE",
            attr="_a_inequality",
        ),
        option(
            "b_inequality",
            str,
            "Type of inequality for the second variable b.",
            default="GE",
            attr="_b_inequality",
        ),
    )

    _a: str
    _b: str
    _residual: str
    _a_inequality: str
    _b_inequality: str
    _sa: float
    _sb: float

    def __post_init__(self) -> None:
        if self._a_inequality not in ("LE", "GE") or self._b_inequality not in ("LE", "GE"):
            raise ValueError("FBComplementarity inequality must be 'LE' or 'GE'")
        # Sign convention: LE → -, GE → + on each term.
        self._sa = -1.0 if self._a_inequality == "LE" else 1.0
        self._sb = -1.0 if self._b_inequality == "LE" else 1.0

    def forward(  # type: ignore[override]
        self,
        a: Scalar,
        b: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        sa, sb = self._sa, self._sb
        eps = torch.finfo(a.dtype).eps
        norm_ab = sqrt(a * a + b * b + eps)
        r = sa * a + sb * b - norm_ab
        if v is None:
            return r
        # ∂r/∂a = sa − a/‖(a,b)‖, ∂r/∂b = sb − b/‖(a,b)‖ — typed Scalar coefficients.
        dr_da = sa - a / norm_ab
        dr_db = sb - b / norm_ab
        return r, self.apply_chain_rule(
            v,
            self._residual,
            {
                self._a: lambda V, c=dr_da: c * V,
                self._b: lambda V, c=dr_db: c * V,
            },
            output=r,
        )


__all__ = ["FBComplementarity"]
