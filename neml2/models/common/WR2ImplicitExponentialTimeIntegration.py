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

"""Python-native mirror of C++ ``common/WR2ImplicitExponentialTimeIntegration.h``."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, derived_input, derived_output, input, option
from ...types import (
    WR2,
    Rot,
    Scalar,
    compose,
    exp_map,
    jvp_compose,
    jvp_exp_map,
)
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("WR2ImplicitExponentialTimeIntegration")
class WR2ImplicitExponentialTimeIntegration(Model):
    r"""Define the implicit discrete exponential time integration residual of a
    rotation variable. The residual can be written as
    $r = s - \exp\left[ (t-t_n)\dot{s}\right] \circ s_n$, where
    $\circ$ denotes the rotation operator.
    """

    hit = HitSchema(
        input("variable", Rot, "Integrated rotation", attr="_var"),
        derived_input("variable", Rot, attr="_var_n", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        derived_input("variable", WR2, attr="_rate", suffix="_rate", override="rate"),
        option("rate", str, "Override name for the variable rate.", default="", attr="_rate_opt"),
        derived_output("variable", Rot, attr="_residual", suffix="_residual"),
    )

    _var: str
    _var_n: str
    _t: str
    _t_n: str
    _rate: str
    _residual: str

    def forward(  # type: ignore[override]
        self,
        s: Rot,
        s_n: Rot,
        t: Scalar,
        t_n: Scalar,
        s_rate: WR2,
        v: ChainRuleDict | None = None,
    ):
        # typed wrapper ops align global ``dt`` (Scalar) against
        # per-crystal ``s_rate``/``s``/``s_n`` (Rot/WR2) automatically — the
        # ``ComposedModel`` preserves sub_batch_ndim across the leaf boundary.
        dt = t - t_n  # Scalar
        scaled = s_rate * dt  # WR2 — auto-aligned
        inc = exp_map(scaled)
        # Residual r = s - compose(inc, s_n)
        composed = compose(inc, s_n)
        r = s - composed
        if v is None:
            return r

        # Differential pushforwards, chained through the typed geometric
        # JVP primitives (jvp_exp_map / jvp_compose) — no leaf-level Jacobians:
        #   ∂r/∂s     = I                         → V
        #   ∂r/∂s~1   = -∂compose/∂s_n            → -jvp_compose(.., dr2=V)
        #   ∂r/∂s_rate: scaled = s_rate·dt        → -jvp_compose(.., dr1=jvp_exp_map(V·dt))
        #   ∂r/∂t:     ∂scaled/∂t = s_rate        → -jvp_compose(.., dr1=jvp_exp_map(s_rate·V))
        #   ∂r/∂t~1:   ∂scaled/∂t~1 = -s_rate     → +jvp_compose(.., dr1=jvp_exp_map(s_rate·V))
        def action_s(V: Rot) -> Rot:
            return V

        def action_sn(V: Rot) -> Rot:
            return -jvp_compose(inc, s_n, dr2=V)

        def action_rate(V: WR2) -> Rot:
            return -jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, V * dt))

        def action_t(V: Scalar) -> Rot:
            return -jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, s_rate * V))

        def action_tn(V: Scalar) -> Rot:
            return jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, s_rate * V))

        actions = {
            self._var: action_s,
            self._var_n: action_sn,
            self._rate: action_rate,
            self._t: action_t,
            self._t_n: action_tn,
        }
        return r, self.apply_chain_rule(v, self._residual, actions, output=r)


__all__ = ["WR2ImplicitExponentialTimeIntegration"]
