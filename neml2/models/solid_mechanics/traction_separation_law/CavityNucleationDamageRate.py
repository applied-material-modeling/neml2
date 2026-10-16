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

"""Cavitated-area-fraction rate contributed by nucleation of new cavities."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("CavityNucleationDamageRate")
class CavityNucleationDamageRate(Model):
    r"""The part of $\dot f$ carried by newly nucleated cavities,

    $$
    \dot{f}_\text{nucleation} = \frac{f}{N} \dot{N}.
    $$

    New cavities arrive at the current mean size, so the area fraction grows
    in proportion to the density.

    Split from
    :class:`~.CavityGrowthDamageRate.CavityGrowthDamageRate` so that a
    nucleation-free variant of the model is assembled by *omitting* leaves
    rather than by toggling an option: drop this model, the
    :class:`~.CavityNucleationRate.CavityNucleationRate`, and the
    `ScalarLinearCombination` that sums the two contributions, hold $N$
    constant, and the remaining composition is the pure cavity-growth model.
    """

    hit = HitSchema(
        input("cavity_fraction", Scalar, "Physical cavitated area fraction"),
        input("cavity_density", Scalar, "Cavity number density per unit boundary area"),
        input("cavity_density_rate", Scalar, "Rate of change of the cavity number density"),
        output(
            "cavity_fraction_nucleation_rate",
            Scalar,
            "Area fraction rate contributed by newly nucleated cavities",
        ),
    )

    def forward(  # type: ignore[override]
        self,
        f: Scalar,
        N: Scalar,
        Ndot: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        fdot = f * Ndot / N

        if v is None:
            return fdot

        df_df = Ndot / N
        df_dN = -fdot / N
        df_dNdot = f / N

        actions: dict[str, ChainRuleAction] = {
            "cavity_fraction": lambda V, c=df_df: c * V,
            "cavity_density": lambda V, c=df_dN: c * V,
            "cavity_density_rate": lambda V, c=df_dNdot: c * V,
        }
        return fdot, self.apply_chain_rule(
            v, "cavity_fraction_nucleation_rate", actions, output=fdot
        )


__all__ = ["CavityNucleationDamageRate"]
