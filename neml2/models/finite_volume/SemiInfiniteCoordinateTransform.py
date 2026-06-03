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

"""Python-native mirror of C++ ``finite_volume/SemiInfiniteCoordinateTransform.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar


@register_native("SemiInfiniteCoordinateTransform")
class SemiInfiniteCoordinateTransform(Model):
    """Transform a semi-infinite coordinate to x / (x + s).

    Linear-fractional map of an input coordinate $x$ into the unit interval
    via the shift parameter $s$. The pushforward in the $x$ direction is
    $s / (x + s)^2 * V$ (the standard d/dx of x/(x+s)); $s$ is a forward-only
    static parameter (the C++ side allows promotion to an nl input, left for a
    follow-on since the native ``ModelUnitTest`` only checks input derivatives).
    """

    hit = HitSchema(
        input("coordinate", Scalar, "Input coordinate."),
        output(
            "transformed_coordinate",
            Scalar,
            "Transformed coordinate.",
            default="state/x_hat",
        ),
        parameter("shift", Scalar, "Shift parameter.", attr="s", allow_nonlinear=True),
    )

    s: Scalar

    def forward(self, *inputs, v: ChainRuleDict | None = None):  # type: ignore[override]
        x = inputs[0]
        s = self._get_param("s", nl_params=(), type_cls=Scalar)

        inv = 1.0 / (x + s)
        x_hat = x * inv

        if v is None:
            return x_hat

        # D-062 pushforward: d(x/(x+s))/dx = s / (x+s)^2. Linear in the tangent
        # V; written as wrapper algebra (no .data, no Jacobian materialization).
        d_x = s * inv * inv

        def x_action(V: Scalar) -> Scalar:
            return d_x * V

        return x_hat, self.apply_chain_rule(
            v,
            "transformed_coordinate",
            {"coordinate": x_action},
            output=x_hat,
        )


__all__ = ["SemiInfiniteCoordinateTransform"]
