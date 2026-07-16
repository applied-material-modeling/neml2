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

"""Python-native R2 exponential-map implicit time integration."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, derived_input, derived_output, input, option
from ...types import (
    R2,
    Scalar,
    exp_map,
    jvp_exp_map,
)
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("R2ImplicitExponentialTimeIntegration")
class R2ImplicitExponentialTimeIntegration(Model):
    r"""Define the implicit discrete exponential time integration residual of a
    full second-order tensor variable. The residual can be written as
    $r = s - \exp\left[ (t-t_n)\dot{s}\right] s_n$, where the exponential is the
    matrix exponential and juxtaposition denotes the matrix product.

    Used for the exponential-map integration of the plastic deformation gradient
    ($F_p = \exp(L_p\,dt)\,F_{p,n}$), where the variable rate $\dot{s}$ is the
    plastic velocity gradient $L_p$. A traceless rate keeps $\det F_p$ constant,
    so isochoric plasticity is preserved exactly.
    """

    hit = HitSchema(
        input("variable", R2, "Integrated tensor", attr="_var"),
        derived_input("variable", R2, attr="_var_n", suffix="~1"),
        input("time", Scalar, "Time", default="t", attr="_t"),
        derived_input("time", Scalar, attr="_t_n", suffix="~1"),
        derived_input("variable", R2, attr="_rate", suffix="_rate", override="rate"),
        option("rate", str, "Override name for the variable rate.", default="", attr="_rate_opt"),
        derived_output("variable", R2, attr="_residual", suffix="_residual"),
    )

    _var: str
    _var_n: str
    _t: str
    _t_n: str
    _rate: str
    _residual: str

    def forward(  # type: ignore[override]
        self,
        s: R2,
        s_n: R2,
        t: Scalar,
        t_n: Scalar,
        s_rate: R2,
        v: ChainRuleDict | None = None,
    ):
        # typed wrapper ops align global ``dt`` (Scalar) against per-crystal
        # ``s_rate``/``s``/``s_n`` (R2) automatically — the ``ComposedModel``
        # preserves sub_batch_ndim across the leaf boundary.
        dt = t - t_n  # Scalar
        scaled = s_rate * dt  # R2 — auto-aligned
        inc = exp_map(scaled)  # R2 matrix exponential
        # Residual r = s - inc @ s_n
        composed = inc @ s_n
        r = s - composed
        if v is None:
            return r

        # Differential pushforwards. The compose ``inc @ s_n`` is bilinear, so
        # its pushforward needs no dedicated JVP primitive — the typed ``@``
        # operator carries the seed-direction axis. The nonlinear piece is
        # ``exp_map``, handled by :func:`jvp_exp_map` (matrix-exp Fréchet):
        #   ∂r/∂s     = I                       → V
        #   ∂r/∂s~1   = -inc                    → -(inc @ V)
        #   ∂r/∂s_rate: scaled = s_rate·dt      → -(jvp_exp_map(scaled, V·dt) @ s_n)
        #   ∂r/∂t:     ∂scaled/∂t = s_rate      → -(jvp_exp_map(scaled, s_rate·V) @ s_n)
        #   ∂r/∂t~1:   ∂scaled/∂t~1 = -s_rate   → +(jvp_exp_map(scaled, s_rate·V) @ s_n)
        def action_s(V: R2) -> R2:
            return V

        def action_sn(V: R2) -> R2:
            return -(inc @ V)

        def action_rate(V: R2) -> R2:
            return -(jvp_exp_map(scaled, V * dt) @ s_n)

        def action_t(V: Scalar) -> R2:
            return -(jvp_exp_map(scaled, s_rate * V) @ s_n)

        def action_tn(V: Scalar) -> R2:
            return jvp_exp_map(scaled, s_rate * V) @ s_n

        actions = {
            self._var: action_s,
            self._var_n: action_sn,
            self._rate: action_rate,
            self._t: action_t,
            self._t_n: action_tn,
        }
        return r, self.apply_chain_rule(v, self._residual, actions, output=r)


__all__ = ["R2ImplicitExponentialTimeIntegration"]
