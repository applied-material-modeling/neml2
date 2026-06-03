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

"""Python-native mirrors of C++ ``common/LinearInterpolation.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import BLOCK_NAME, HitSchema, input, output, parameter
from ...types import (
    Scalar,
    jvp_linear_interpolation,
    linear_interpolation,
)


@register_native("ScalarLinearInterpolation")
class ScalarLinearInterpolation(Model):
    """Interpolate a Scalar as a function of the given argument. See
    neml2::Interpolation for rules on shapes of the interpolant and the
    argument. This object performs a _linear interpolation_.
    """

    # Piecewise-linear interp has constant slope between knots; the second
    # derivative w.r.t. the argument is structurally zero. We can therefore
    # safely opt in to the v2/vh path so a wrapping Normality can compose
    # through this leaf without tripping the SUPPORTS_SECOND_ORDER guard.
    SUPPORTS_SECOND_ORDER = True

    # C++ ``Interpolation`` defaults its output variable to the model's block
    # name; ``output(..., default=BLOCK_NAME)`` reproduces that.
    hit = HitSchema(
        input("argument", Scalar, "Argument used to query the interpolant", attr="_argument"),
        output(
            "output",
            Scalar,
            "Scalar output of the interpolant. If not specified, "
            "the object name will be used as the output name.",
            default=BLOCK_NAME,
            attr="_output",
        ),
        parameter("abscissa", Scalar, "Scalar defining the abscissa values of the interpolant"),
        parameter("ordinate", Scalar, "Scalar defining the ordinate values of the interpolant"),
    )

    # Class-level type hints so pyright sees the re-wrapped types from
    # Model.__getattr__ (which otherwise infers `Module` from nn.Module's
    # registered-parameter accessor).
    abscissa: Scalar
    ordinate: Scalar
    _argument: str
    _output: str

    def forward(  # type: ignore[override]
        self,
        *inputs: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        if len(inputs) != 1:
            raise ValueError(f"ScalarLinearInterpolation expected 1 input, got {len(inputs)}")
        x_arg = inputs[0]
        out = linear_interpolation(x_arg, self.abscissa, self.ordinate)
        if v is None:
            return out

        # Differential pushforward via the typed JVP primitive — hides the
        # ``searchsorted`` + gather indexing behind a typed-function boundary
        # so the leaf body stays in pure typed-wrapper algebra.
        def argument_action(V: Scalar) -> Scalar:
            return jvp_linear_interpolation(x_arg, self.abscissa, self.ordinate, V)

        actions_1 = {self._argument: argument_action}

        if v2 is None and vh is None:
            return out, self.apply_chain_rule(v, self._output, actions_1, output=out)

        # Second-order pushforward. For
        # piecewise-linear interp the Hessian w.r.t. the argument is
        # structurally zero between knots, so action_2 returns a zero
        # bilinear shaped like ``Va * Vb`` (the framework still iterates
        # the seed-pair outer; an all-zeros result is what propagate_tangents
        # contributes to the second-order accumulator).
        actions_2 = {(self._argument, self._argument): lambda Va, Vb: 0.0 * (Va * Vb)}

        return out, *self.propagate_tangents(
            v, self._output, actions_1, output=out, v2=v2, actions_2=actions_2, vh=vh
        )


__all__ = ["ScalarLinearInterpolation"]
