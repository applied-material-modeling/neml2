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

"""Salehani-Irani cohesive traction assembly."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, Vec, vec_from_scalars
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SalehaniIraniTraction")
class SalehaniIraniTraction(Model):
    r"""Assemble Salehani-Irani traction from an externally updated damage."""

    hit = HitSchema(
        input("damage", Scalar, "Current damage"),
        input("normal_separation", Scalar, "Normal opening separation"),
        input(
            "normal_penetration",
            Scalar,
            "Optional normal penetration included through the penalty term",
            default=None,
            attr="_dn_pen_name",
        ),
        input("tangential_separation_1", Scalar, "First tangential separation"),
        input("tangential_separation_2", Scalar, "Second tangential separation"),
        output("traction", Vec, "Traction vector"),
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
        parameter("normal_strength", Scalar, "Normal strength", attr="Tmax_n"),
        parameter("shear_strength", Scalar, "Shear strength", attr="Tmax_t"),
        parameter("penalty_stiffness", Scalar, "Penalty stiffness", attr="Kpen", default="0"),
    )

    delta_u0_n: Scalar
    delta_u0_t: Scalar
    Tmax_n: Scalar
    Tmax_t: Scalar
    Kpen: Scalar
    _dn_pen_name: str | None

    def forward(  # type: ignore[override]
        self,
        *args: Scalar,
        v: ChainRuleDict | None = None,
    ):
        names = list(self.input_spec)
        bound = dict(zip(names, args, strict=True))
        renames = getattr(self, "_var_renames", {})

        def resolved(name: str) -> str:
            return renames.get(name, name)

        damage = bound[resolved("damage")]
        dn = bound[resolved("normal_separation")]
        ds1 = bound[resolved("tangential_separation_1")]
        ds2 = bound[resolved("tangential_separation_2")]
        dn_pen = bound.get(self._dn_pen_name) if self._dn_pen_name is not None else None
        delta_n = self._get_param("delta_u0_n", (), Scalar)
        delta_t = math.sqrt(2.0) * self._get_param("delta_u0_t", (), Scalar)
        an = math.e * self._get_param("Tmax_n", (), Scalar)
        at = math.sqrt(2.0 * math.e) * self._get_param("Tmax_t", (), Scalar)
        Kpen = self._get_param("Kpen", (), Scalar)
        bn = dn / delta_n
        bs1 = ds1 / delta_t
        bs2 = ds2 / delta_t
        factor = 1.0 - damage
        Tn = an * bn * factor
        if dn_pen is not None:
            Tn = Tn + Kpen * dn_pen
        traction = vec_from_scalars(Tn, at * bs1 * factor, at * bs2 * factor)
        if v is None:
            return traction

        zero = Scalar.from_value(0.0, like=damage)

        def diagonal(component: int, coefficient: Scalar):
            def action(V: Scalar) -> Vec:
                values = [zero * V, zero * V, zero * V]
                values[component] = coefficient * V
                return vec_from_scalars(*values)

            return action

        actions: dict[str, ChainRuleAction] = {
            "damage": lambda V: vec_from_scalars(-an * bn * V, -at * bs1 * V, -at * bs2 * V),
            "normal_separation": diagonal(0, an * factor / delta_n),
            "tangential_separation_1": diagonal(1, at * factor / delta_t),
            "tangential_separation_2": diagonal(2, at * factor / delta_t),
        }
        if self._dn_pen_name is not None:
            actions[self._dn_pen_name] = diagonal(0, Kpen)
        return traction, self.apply_chain_rule(v, "traction", actions, output=traction)


__all__ = ["SalehaniIraniTraction"]
