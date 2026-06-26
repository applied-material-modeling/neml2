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

"""Python-native mirror of the C++ ``HermiteSmoothStep`` model."""

from __future__ import annotations

import torch

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output, parameter
from ...types import Scalar, clamp
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("HermiteSmoothStep")
class HermiteSmoothStep(Model):
    r"""The smooth step function defined by Hermite polynomials.

    For $x$ mapped onto the unit interval via $u = (x - x0) / (x1 - x0)$
    and clamped to ``[eps, 1 - eps]``, the forward is the cubic Hermite step
    $y = 3 u^2 - 2 u^3$ (or its complement ``1 - y`` when ``complement = true``).
    The derivative $dy/dx = 6 u (1 - u) / (x1 - x0)$ vanishes naturally at
    the clamp boundaries, mirroring the C++ ``set_value`` implementation.
    """

    hit = HitSchema(
        input("argument", Scalar, "Argument of the smooth step function"),
        output("value", Scalar, "Value of the smooth step function"),
        parameter("lower_bound", Scalar, "Lower bound of the argument", attr="x0"),
        parameter("upper_bound", Scalar, "Upper bound of the argument", attr="x1"),
        option(
            "complement",
            bool,
            "Whether to take the complement of the smooth step function, i.e. 1 - smooth_step",
            default=False,
            attr="_complement",
        ),
    )

    # ``from_hit`` auto-declares ``lower_bound`` (as ``x0``) and ``upper_bound``
    # (as ``x1``). The ``complement`` option lands on ``self._complement`` via
    # ``attr=``. Annotate so pyright sees the typed wrappers returned by
    # ``Model.__getattr__`` instead of nn.Module's ``Module`` hint.
    x0: Scalar
    x1: Scalar
    _complement: bool

    def forward(  # type: ignore[override]
        self,
        argument: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        x0 = self._get_param("x0", promoted_params, Scalar)
        x1 = self._get_param("x1", promoted_params, Scalar)
        eps = torch.finfo(argument.dtype).eps
        dx = x1 - x0
        u = clamp((argument - x0) / dx, eps, 1.0 - eps)
        y = 3.0 * u * u - 2.0 * u * u * u
        out = 1.0 - y if self._complement else y
        if v is None:
            return out

        # Differential pushforward: dy/dx = 6 u (1 - u) / (x1 - x0). The
        # clamp's flat tails ensure this naturally vanishes at the boundaries
        # via ``(1 - u) * u == 0`` at ``u = 0`` and ``u = 1``. Negate when the
        # forward took the complement.
        dy_dx = 6.0 * u * (1.0 - u) / dx
        dout_dx = -dy_dx if self._complement else dy_dx
        return out, self.apply_chain_rule(
            v,
            "value",
            {"argument": lambda V, c=dout_dx: c * V},
            output=out,
        )


__all__ = ["HermiteSmoothStep"]
