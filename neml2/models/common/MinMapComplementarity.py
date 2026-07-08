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

"""Hard-switch (minimum-map) KKT complementarity condition."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, option, output
from ...types import Scalar, lt, where
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("MinMapComplementarity")
class MinMapComplementarity(Model):
    r"""KKT complementarity condition enforced with the *minimum map*.

    The complementarity conditions $a \ge 0, b \ge 0, ab = 0$ are equivalent to
    the single scalar equation
    $$r = \min(a, b) = 0.$$
    The residual is piecewise-linear: it equals whichever of the two arguments
    is currently smaller, a hard switch that gives semismooth-Newton /
    active-set behavior and converges robustly in return mapping. The switch is
    selected per batch entry with ``where`` (no Python branching), so the model
    lowers through AOTI and runs on GPU; the trade-off is a piecewise-constant
    Jacobian that is non-smooth at the switch $a = b$.

    The sign of each argument follows its declared inequality: ``LE`` enforces
    $\le 0$ (coefficient $-1$) and ``GE`` enforces $\ge 0$ (coefficient $+1$),
    so the enforced residual is $r = \min(s_a\, a,\; s_b\, b)$.
    """

    hit = HitSchema(
        input("a", Scalar, "First condition", attr="_a"),
        input("b", Scalar, "Second condition", attr="_b"),
        # Renaming the output (e.g. ``complementarity = 'd_residual'``) lets the
        # host NonlinearSystem's ``<unknown>_residual`` inference wire this leaf
        # up as a residual out of the box.
        output(
            "complementarity",
            Scalar,
            "The minimum-map complementarity condition",
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
            raise ValueError("MinMapComplementarity inequality must be 'LE' or 'GE'")
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
        ta = sa * a
        tb = sb * b
        # ``lt`` is strict, so a tie (ta == tb, measure zero) resolves to the
        # b-branch; the primal and Jacobian below share this single mask so the
        # returned Jacobian is exactly the derivative of the selected branch.
        mask = lt(ta, tb)
        r = where(mask, ta, tb)  # r = min(sa * a, sb * b)
        if v is None:
            return r
        # The residual equals whichever branch is active, so each input's
        # tangent passes straight through (times its sign) on its own branch and
        # vanishes on the other. Apply the same hard switch directly to the
        # incoming tangent V rather than forming a coefficient and multiplying.
        return r, self.apply_chain_rule(
            v,
            self._residual,
            {
                self._a: lambda V, m=mask: where(m, sa * V, Scalar.zeros_like(V)),
                self._b: lambda V, m=mask: where(m, Scalar.zeros_like(V), sb * V),
            },
            output=r,
        )


__all__ = ["MinMapComplementarity"]
