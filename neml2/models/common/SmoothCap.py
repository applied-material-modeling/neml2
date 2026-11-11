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
"""Smooth upper cap on a positive scalar."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output
from ...types import Scalar, pow
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("SmoothCap")
class SmoothCap(Model):
    r"""Smoothly cap a strictly positive scalar from above,
    $y = \left( x^{-p} + x_{max}^{-p} \right)^{-1/p}$.

    This is a smooth minimum of $x$ and $x_{max}$: $y < \min(x, x_{max})$
    everywhere, $y \to x$ for $x \ll x_{max}$ and $y \to x_{max}$ for
    $x \gg x_{max}$. Larger $p$ gives a sharper corner. A typical use is
    bounding a temperature-dependent rate sensitivity (e.g. from
    ``KocksMeckingRateSensitivity``) before it reaches a power-law flow rule,
    where very large exponents overflow at low temperature.

    The argument must be strictly positive.
    """

    hit = HitSchema(
        input("argument", Scalar, "Argument to cap; must be strictly positive"),
        output("value", Scalar, "Smoothly capped value"),
        option("maximum", float, "Cap value approached for large arguments", attr="_xmax"),
        option(
            "exponent",
            float,
            "Smoothing exponent; larger values give a sharper transition",
            default=4.0,
            attr="_p",
        ),
    )

    _xmax: float
    _p: float

    def forward(  # type: ignore[override]
        self,
        x: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        p = self._p
        y = pow(pow(x, -p) + self._xmax ** (-p), -1.0 / p)

        if v is None:
            return y

        # dy/dx = x^(-p-1) * (x^(-p) + xmax^(-p))^(-1/p - 1) = (y / x)^(p + 1)
        dy_dx = pow(y / x, p + 1.0)
        return y, self.apply_chain_rule(
            v, "value", {"argument": lambda V, c=dy_dx: c * V}, output=y
        )


__all__ = ["SmoothCap"]
