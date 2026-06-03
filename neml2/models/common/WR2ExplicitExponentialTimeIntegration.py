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

"""Python-native mirror of C++ ``common/WR2ExplicitExponentialTimeIntegration.h``."""

from __future__ import annotations

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, derived_input, input, option, output
from ...types import (
    WR2,
    Rot,
    Scalar,
    compose,
    exp_map,
    jvp_compose,
    jvp_exp_map,
)


@register_native("WR2ExplicitExponentialTimeIntegration")
class WR2ExplicitExponentialTimeIntegration(Model):
    r"""Perform explicit discrete exponential time integration of a rotation. The
    update can be written as $s = \exp\left[ (t-t_n)\dot{s}\right] \circ s_n$,
    where $\circ$ denotes the rotation operator.
    """

    hit = HitSchema(
        output("variable", Rot, "Variable being integrated", attr="_var"),
        derived_input("variable", Rot, attr="_var_n", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        derived_input("variable", WR2, attr="_rate", suffix="_rate", override="rate"),
        option("rate", str, "Name of the variable rate.", default="", attr="_rate_opt"),
    )

    _var: str
    _var_n: str
    _t: str
    _t_n: str
    _rate: str

    def forward(  # type: ignore[override]
        self,
        s_n: Rot,
        t: Scalar,
        t_n: Scalar,
        s_rate: WR2,
        v: ChainRuleDict | None = None,
    ):
        # typed wrapper ops auto-align global ``dt`` (Scalar) against
        # per-crystal ``s_rate``/``s_n`` (Rot/WR2) — the ``ComposedModel`` preserves
        # sub_batch_ndim across the leaf boundary.
        dt = t - t_n  # Scalar
        scaled = s_rate * dt  # WR2 — auto-aligned
        inc = exp_map(scaled)
        # Update s = compose(inc, s_n) — apply the incremental rotation ``inc`` to
        # the previous rotation ``s_n`` (C++ ``_sn().rotate(inc)``).
        s = compose(inc, s_n)
        if v is None:
            return s

        # Differential pushforwards, chained through the typed geometric
        # JVP primitives (jvp_exp_map / jvp_compose) — no leaf-level Jacobians:
        #   ds/ds~1    = ∂compose/∂s_n           → jvp_compose(.., dr2=V)
        #   ds/ds_rate : scaled = s_rate·dt      → jvp_compose(.., dr1=jvp_exp_map(V·dt))
        #   ds/dt      : ∂scaled/∂t = s_rate     → jvp_compose(.., dr1=jvp_exp_map(s_rate·V))
        #   ds/dt~1    : ∂scaled/∂t~1 = -s_rate  → -jvp_compose(.., dr1=jvp_exp_map(s_rate·V))
        def action_sn(V: Rot) -> Rot:
            return jvp_compose(inc, s_n, dr2=V)

        def action_rate(V: WR2) -> Rot:
            return jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, V * dt))

        def action_t(V: Scalar) -> Rot:
            return jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, s_rate * V))

        def action_tn(V: Scalar) -> Rot:
            return -jvp_compose(inc, s_n, dr1=jvp_exp_map(scaled, s_rate * V))

        actions = {
            self._var_n: action_sn,
            self._rate: action_rate,
            self._t: action_t,
            self._t_n: action_tn,
        }
        return s, self.apply_chain_rule(v, self._var, actions, output=s)


__all__ = ["WR2ExplicitExponentialTimeIntegration"]
