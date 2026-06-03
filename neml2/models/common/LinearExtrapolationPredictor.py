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

"""Python-native mirror of C++ ``common/LinearExtrapolationPredictor.h``."""

from __future__ import annotations

from typing import cast

import torch

from ...chain_rule import ChainRuleDict
from ...factory import register_native
from ...schema import HitSchema, option
from ...types import (
    R2,
    SR2,
    Rot,
    Scalar,
    TensorWrapper,
    gt,
    where,
)
from ...types import (
    abs as wrap_abs,
)
from .._hit import _opt_list_str
from .ConstantExtrapolationPredictor import ConstantExtrapolationPredictor


def _read_list_str(node, name):  # noqa: ANN001, ANN202
    return list(node.param_list_str(name))


def _read_str(node, name):  # noqa: ANN001, ANN202
    return node.param_str(name)


def _opt_str(node, name, default):  # noqa: ANN001, ANN202
    return node.param_str(name) if node.find(name) is not None else default


@register_native("LinearExtrapolationPredictor")
class LinearExtrapolationPredictor(ConstantExtrapolationPredictor):
    r"""Use temporal extrapolation assuming constant rate of change as the initial guess for the
    unknowns at the current time step. The linear extrapolation can be written as
    $u = u_n + (u_n - u_{n-1}) \frac{t - t_n}{t_n - t_{n-1}}$, where $u$ is the unknown
    and $n$ is the time step counter, respectively.
    """

    hit = HitSchema(
        option(
            "time",
            str,
            "Time",
            default="t",
            reader=_read_str,
            optional_reader=_opt_str,
        ),
        option(
            "unknowns_SR2",
            list,
            "The unknowns to extrapolate of type SR2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_Scalar",
            list,
            "The unknowns to extrapolate of type Scalar",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_Rot",
            list,
            "The unknowns to extrapolate of type Rot",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
        option(
            "unknowns_R2",
            list,
            "The unknowns to extrapolate of type R2",
            default=[],
            reader=_read_list_str,
            optional_reader=_opt_list_str,
        ),
    )

    def __init__(
        self,
        unknowns_SR2: list[str],
        unknowns_Scalar: list[str],
        unknowns_Rot: list[str] | None = None,
        unknowns_R2: list[str] | None = None,
        time: str = "t",
    ) -> None:
        # Defer parent __init__ to fill in _sr2 / _scalar / _rot / _r2 and the
        # (var~1 -> var) spec, then extend the input spec with the time history
        # (t, t~1, t~2) and the (var~2) inputs that linear extrapolation needs.
        super().__init__(
            unknowns_SR2=unknowns_SR2,
            unknowns_Scalar=unknowns_Scalar,
            unknowns_Rot=unknowns_Rot,
            unknowns_R2=unknowns_R2,
        )
        self._time = time
        # Positional ordering for forward(): keep parent's (var~1) inputs first,
        # then time triple, then (var~2) for every unknown — same per-type
        # ordering as parent so var_n[i] and var_nm1[i] line up.
        extra_in: dict[str, type[TensorWrapper]] = {
            time: Scalar,
            f"{time}~1": Scalar,
            f"{time}~2": Scalar,
            **{f"{u}~2": SR2 for u in self._sr2},
            **{f"{u}~2": Scalar for u in self._scalar},
            **{f"{u}~2": Rot for u in self._rot},
            **{f"{u}~2": R2 for u in self._r2},
        }
        self.input_spec = {**self.input_spec, **extra_in}

    def forward(  # type: ignore[override]
        self,
        *inputs: TensorWrapper,
        v: ChainRuleDict | None = None,
    ):
        # Unpack inputs in the order declared by input_spec:
        #   [var_n_SR2..., var_n_Scalar..., var_n_Rot...,
        #    t, t_n, t_nm1,
        #    var_nm1_SR2..., var_nm1_Scalar..., var_nm1_Rot...]
        n_sr2 = len(self._sr2)
        n_scalar = len(self._scalar)
        n_rot = len(self._rot)
        n_r2 = len(self._r2)
        n_n = n_sr2 + n_scalar + n_rot + n_r2

        var_n: list[TensorWrapper] = list(inputs[:n_n])
        t_val = cast(Scalar, inputs[n_n])
        t_n = cast(Scalar, inputs[n_n + 1])
        t_nm1 = cast(Scalar, inputs[n_n + 2])
        var_nm1: list[TensorWrapper] = list(inputs[n_n + 3 : 2 * n_n + 3])

        # Per the C++ predict(): on the first step (or whenever |t_n - t_nm1|
        # is at machine precision) fall back to ``var_n``; otherwise extrapolate.
        # gt + where both broadcast a Scalar condition against any wrapper type.
        dt_nm1 = t_n - t_nm1  # Scalar
        eps = torch.finfo(t_val.dtype).eps
        cond = gt(wrap_abs(dt_nm1), eps)

        # ratio = (t - t_n) / (t_n - t_nm1); Scalar / Scalar handles alignment.
        # We avoid dividing by a possibly-zero dt_nm1 in the chain-rule branch
        # by selecting through ``where`` -- but the forward itself reads ratio
        # before the where mask, so we must guard against NaN/Inf there too.
        # Replace dt_nm1 by 1 where the condition is inactive so the division
        # never produces NaN/Inf in the active branch's value (and the inactive
        # branch is then masked out by ``where``).
        one = Scalar.from_value(1.0, like=t_val)
        safe_dt_nm1 = where(cond, dt_nm1, one)
        ratio = (t_val - t_n) / safe_dt_nm1  # Scalar

        outs: list[TensorWrapper] = []
        for u_n, u_nm1 in zip(var_n, var_nm1, strict=True):
            # u_extrap = u_n + (u_n - u_nm1) * ratio. ``ratio`` is a Scalar;
            # mixed Scalar * SR2/Rot/Scalar dispatches via the wrapper's __mul__.
            delta = u_n - u_nm1
            u_extrap = u_n + delta * ratio
            outs.append(where(cond, u_extrap, u_n))

        if v is None:
            if len(outs) == 1:
                return outs[0]
            return tuple(outs)

        # D-062 pushforward. Each output depends on var~1, var~2, t, t~1, t~2.
        # On the active (cond=True) branch:
        #   du/d(var~1)  = (1 + ratio) · I        -> (1 + ratio) · V
        #   du/d(var~2)  = -ratio · I             -> -ratio · V
        #   du/d(t)      = (u_n - u_nm1) / dt_nm1
        #   du/d(t~1)    = -(u_n - u_nm1)·(t - t_nm1)/dt_nm1^2
        #     because d(ratio)/d(t_n) = -1/dt_nm1 + (t - t_n)·(-1)/dt_nm1·(-1)
        #                              = -1/dt_nm1 - (t - t_n)/dt_nm1^2
        #                              = -(t - t_nm1)/dt_nm1^2
        #   du/d(t~2)    = (u_n - u_nm1) · ratio / dt_nm1
        #     because d(ratio)/d(t_nm1) = (t - t_n) / dt_nm1^2 = ratio / dt_nm1
        # On the inactive branch (|dt_nm1| <= eps), the predictor reduces to u_n:
        #   du/d(var~1) = I; everything else is structural zero.
        inv_dt = one / safe_dt_nm1
        one_plus_ratio = ratio + 1.0
        neg_ratio = -ratio
        # (t - t_nm1) * inv_dt^2
        d_ratio_dt_n = -((t_val - t_nm1) * inv_dt * inv_dt)
        d_ratio_dt_nm1 = ratio * inv_dt

        in_names = list(self.input_spec)
        var_n_names = in_names[:n_n]
        t_name = in_names[n_n]
        t_n_name = in_names[n_n + 1]
        t_nm1_name = in_names[n_n + 2]
        var_nm1_names = in_names[n_n + 3 : 2 * n_n + 3]
        out_names = list(self.output_spec)

        v_out: ChainRuleDict = {}
        for i, out_name in enumerate(out_names):
            u_n_i = var_n[i]
            u_nm1_i = var_nm1[i]
            delta_i = u_n_i - u_nm1_i

            def make_var_n_action():  # captures cond, one_plus_ratio
                def action(V: TensorWrapper) -> TensorWrapper:
                    return where(cond, one_plus_ratio * V, V)

                return action

            def make_var_nm1_action():
                def action(V: TensorWrapper) -> TensorWrapper:
                    zero_v = V * Scalar.from_value(0.0, like=t_val)
                    return where(cond, neg_ratio * V, zero_v)

                return action

            def make_t_action(d: TensorWrapper = delta_i):
                def action(V: Scalar) -> TensorWrapper:
                    active = d * (inv_dt * V)
                    zero_v = d * Scalar.from_value(0.0, like=t_val)
                    return where(cond, active, zero_v)

                return action

            def make_t_n_action(d: TensorWrapper = delta_i):
                def action(V: Scalar) -> TensorWrapper:
                    active = d * (d_ratio_dt_n * V)
                    zero_v = d * Scalar.from_value(0.0, like=t_val)
                    return where(cond, active, zero_v)

                return action

            def make_t_nm1_action(d: TensorWrapper = delta_i):
                def action(V: Scalar) -> TensorWrapper:
                    active = d * (d_ratio_dt_nm1 * V)
                    zero_v = d * Scalar.from_value(0.0, like=t_val)
                    return where(cond, active, zero_v)

                return action

            actions = {
                var_n_names[i]: make_var_n_action(),
                var_nm1_names[i]: make_var_nm1_action(),
                t_name: make_t_action(),
                t_n_name: make_t_n_action(),
                t_nm1_name: make_t_nm1_action(),
            }
            # apply_chain_rule returns {ext_output: {leaf: contribution, ...}};
            # merge into the cross-output v_out.
            v_out.update(self.apply_chain_rule(v, out_name, actions, output=outs[i]))

        if len(outs) == 1:
            return outs[0], v_out
        return (*outs, v_out)


__all__ = ["LinearExtrapolationPredictor"]
