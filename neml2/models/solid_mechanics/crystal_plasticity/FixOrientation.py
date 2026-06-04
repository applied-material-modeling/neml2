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

"""Python-native mirror of the C++ ``FixOrientation`` crystal-plasticity leaf."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, option, output
from ....types import Rot, inner, lt, where


@register_neml2_object("FixOrientation")
class FixOrientation(Model):
    r"""Swap the modified-Rodrigues representation to the shadow parameter set
    when $\left\lVert r \right\rVert^2 \geq t$ (default threshold
    $t = 1$), avoiding the $2\pi$ singularity. The shadow map is
    $r^* = -r / \left\lVert r \right\rVert^2$.

    See Banks, Matthew Jarrett, "Switching Methods for Three-Dimensional
    Rotational Dynamics Using Modified Rodrigues Parameters." (2023).
    """

    hit = HitSchema(
        input("input", Rot, "Name of input tensor of orientations to operate on."),
        output("output", Rot, "Name of output tensor"),
        option(
            "threshold",
            float,
            "Threshold value for translating to the shadow parameters",
            default=1.0,
            attr="_threshold",
        ),
    )

    # ``from_hit`` stores the ``threshold`` option on ``self._threshold``
    # via ``attr=``. Annotate so pyright sees a plain ``float`` rather than
    # ``nn.Module``'s generic ``Module`` hint.
    _threshold: float

    def forward(  # type: ignore[override]
        self,
        r: Rot,
        v: ChainRuleDict | None = None,
    ) -> Rot | tuple[Rot, ChainRuleDict]:
        # ns = ‖r‖² as a Scalar; ``inner(Rot, Rot)`` contracts the three MRP
        # components into a Scalar over the shared batch + sub-batch axes.
        ns = inner(r, r)
        # Shadow MRP set: r* = -r / ns. Same orientation, but with norm 1/‖r‖,
        # which falls back inside the unit ball whenever ‖r‖ > 1.
        shadow = -r / ns
        cond = lt(ns, self._threshold)
        out = where(cond, r, shadow)
        if v is None:
            return out

        # D-062 pushforward.
        #   ‖r‖ < t : f(r) = r              ⇒ d f · V = V
        #   ‖r‖ ≥ t : f(r) = -r / (r·r)     ⇒ d f · V = (2 (r·V) r - (r·r) V) / (r·r)²
        # Both branches stay in typed wrapper algebra; ``where`` selects the
        # per-element tangent — no Jacobian materialised.
        ns_sq = ns * ns

        def r_action(V: Rot) -> Rot:
            vdotr = inner(r, V)
            d_shadow_v = (vdotr * r) * (2.0 / ns_sq) - V / ns
            return where(cond, V, d_shadow_v)

        return out, self.apply_chain_rule(
            v,
            "output",
            {"input": r_action},
            output=out,
        )
