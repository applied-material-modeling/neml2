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

"""Python-native mirror of the C++ ``SymmetricHermiteInterpolation`` model."""

from __future__ import annotations

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import BLOCK_NAME, HitSchema, input, output, parameter
from ...types import Scalar, clamp, lt, where


@register_native("SymmetricHermiteInterpolation")
class SymmetricHermiteInterpolation(Model):
    r"""Define the symmetric Hermite interpolation function, taking the form of
    $\dfrac{1}{x_h-x_l}(24c^2-32c^3)$ for $0 le c le 0.5$;
    $\dfrac{1}{x_h-x_l} (24(1-c)^2 - 32(1-c)^3)$ for $0.5 le c le 1$,
    and 0.0 otherwise. Here, $c = \frac{x-x_l}{x_h-x_l}$ where $x_l$
    and $x_h$ are the lower and upper bound for rescaling the input
    argument.
    """

    hit = HitSchema(
        input("argument", Scalar, "Argument of the smooth step function"),
        output(
            "output",
            Scalar,
            "Value of the smooth step function. If not specified, the object "
            "name will be used as the output name.",
            default=BLOCK_NAME,
        ),
        parameter("lower_bound", Scalar, "Lower bound of the argument", attr="x0"),
        parameter("upper_bound", Scalar, "Upper bound of the argument", attr="x1"),
    )

    # ``from_hit`` auto-declares ``lower_bound`` (as ``x0``) and ``upper_bound``
    # (as ``x1``). Annotate the members so pyright sees the typed wrappers
    # ``Model.__getattr__`` returns instead of nn.Module's ``Module`` hint.
    x0: Scalar
    x1: Scalar

    def forward(  # type: ignore[override]
        self,
        argument: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        x0 = self._get_param("x0", nl_params, Scalar)
        x1 = self._get_param("x1", nl_params, Scalar)
        eps = torch.finfo(argument.dtype).eps
        dx = x1 - x0
        scale = 1.0 / dx
        # Rescale the input onto the unit interval and clamp away from the
        # endpoints so the derivative is well-defined at the boundaries
        # (matches the C++ ``clamp((_x - _x0) / (_x1 - _x0), eps, 1 - eps)``).
        c = clamp((argument - x0) / dx, eps, 1.0 - eps)

        # Forward: piecewise cubic Hermite, symmetric around c = 0.5.
        omc = 1.0 - c
        f_xl = 24.0 * c * c - 32.0 * c * c * c
        f_xh = 24.0 * omc * omc - 32.0 * omc * omc * omc
        mask = lt(c, 0.5)
        y = where(mask, scale * f_xl, scale * f_xh)
        if v is None:
            return y

        # D-062 pushforward: chain rule through ``c = (x - x0) / dx``. With
        # ``dc/dx = scale``, ``dy/dx = scale^2 * df/dc`` where df/dc is the
        # branch-selected polynomial derivative below. The clamp's flat tails
        # ensure the derivative naturally vanishes outside ``[eps, 1 - eps]``
        # (``f'(0) = f'(1) = 0`` for both branches).
        df_xl = 48.0 * c - 96.0 * c * c
        df_xh = -48.0 * omc + 96.0 * omc * omc
        dy_dx = scale * scale * where(mask, df_xl, df_xh)

        return y, self.apply_chain_rule(
            v,
            "output",
            {"argument": lambda V, c=dy_dx: c * V},
            output=y,
        )


__all__ = ["SymmetricHermiteInterpolation"]
