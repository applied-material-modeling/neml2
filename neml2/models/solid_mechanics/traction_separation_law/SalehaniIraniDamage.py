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

"""Salehani-Irani cohesive damage envelope."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, exp
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("SalehaniIraniDamage")
class SalehaniIraniDamage(Model):
    r"""Trial damage $1-\exp[-(b_n+b_{s1}^2+b_{s2}^2)]$."""

    hit = HitSchema(
        input("normal_separation", Scalar, "Normal separation"),
        input("tangential_separation_1", Scalar, "First tangential separation"),
        input("tangential_separation_2", Scalar, "Second tangential separation"),
        output("trial_damage", Scalar, "Rate-independent trial damage"),
        parameter(
            "normal_characteristic_length",
            Scalar,
            "Normal characteristic length",
            attr="delta_u0_n",
        ),
        parameter(
            "tangential_characteristic_length",
            Scalar,
            "Tangential characteristic length",
            attr="delta_u0_t",
        ),
    )

    delta_u0_n: Scalar
    delta_u0_t: Scalar

    def forward(  # type: ignore[override]
        self,
        dn: Scalar,
        ds1: Scalar,
        ds2: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ):
        delta_n = self._get_param("delta_u0_n", promoted_params, Scalar)
        delta_t = math.sqrt(2.0) * self._get_param("delta_u0_t", promoted_params, Scalar)
        bn = dn / delta_n
        bs1 = ds1 / delta_t
        bs2 = ds2 / delta_t
        decay = exp(-(bn + bs1 * bs1 + bs2 * bs2))
        damage = 1.0 - decay
        if v is None:
            return damage

        return damage, self.apply_chain_rule(
            v,
            "trial_damage",
            {
                "normal_separation": lambda V: decay / delta_n * V,
                "tangential_separation_1": lambda V: decay * 2.0 * ds1 / (delta_t * delta_t) * V,
                "tangential_separation_2": lambda V: decay * 2.0 * ds2 / (delta_t * delta_t) * V,
            },
            output=damage,
        )


__all__ = ["SalehaniIraniDamage"]
