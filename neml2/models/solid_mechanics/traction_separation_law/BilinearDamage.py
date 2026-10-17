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

"""Bilinear cohesive damage envelope."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, lt, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("BilinearDamage")
class BilinearDamage(Model):
    r"""Bilinear trial damage as a function of effective separation."""

    hit = HitSchema(
        input("effective_separation", Scalar, "Effective separation"),
        output("trial_damage", Scalar, "Rate-independent trial damage"),
        parameter(
            "critical_separation",
            Scalar,
            "Critical damage-onset separation",
            attr="delta_c",
            allow_promotion=True,
        ),
        parameter(
            "full_separation",
            Scalar,
            "Full failure separation",
            attr="delta_f",
            allow_promotion=True,
        ),
    )

    delta_c: Scalar
    delta_f: Scalar

    def forward(  # type: ignore[override]
        self,
        delta_m: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        delta_c = self._get_param("delta_c", promoted_params, Scalar)
        delta_f = self._get_param("delta_f", promoted_params, Scalar)
        one = Scalar.from_value(1.0, like=delta_m)
        zero = Scalar.from_value(0.0, like=delta_m)
        diff = delta_f - delta_c
        safe_diff = where(gt(diff, 0.0), diff, one)
        after_onset = gt(delta_m, delta_c)
        safe_delta_m = where(after_onset, delta_m, one)
        interior_damage = delta_f * (delta_m - delta_c) / (safe_delta_m * safe_diff)
        before_failure = lt(delta_m, delta_f)
        damage = where(
            lt(delta_m, delta_c),
            zero,
            where(before_failure, interior_damage, one),
        )
        if v is None:
            return damage

        def interior_action(value: Scalar, V: Scalar) -> Scalar:
            return where(
                after_onset,
                where(before_failure, value * V, Scalar.zeros_like(V)),
                Scalar.zeros_like(V),
            )

        inv_dm = one / delta_m
        inv_diff = one / safe_diff
        inv_diff_sq = inv_diff * inv_diff
        actions: dict[str, ChainRuleAction] = {
            "effective_separation": lambda V: interior_action(
                delta_f * delta_c * inv_dm * inv_dm * inv_diff, V
            )
        }
        critical = self._promoted_params.get("delta_c")
        if critical is not None:
            actions[critical.input_name] = lambda V: interior_action(
                delta_f * (delta_m - delta_f) * inv_dm * inv_diff_sq, V
            )
        full = self._promoted_params.get("delta_f")
        if full is not None:
            actions[full.input_name] = lambda V: interior_action(
                -delta_c * (delta_m - delta_c) * inv_dm * inv_diff_sq, V
            )
        return damage, self.apply_chain_rule(v, "trial_damage", actions, output=damage)


__all__ = ["BilinearDamage"]
