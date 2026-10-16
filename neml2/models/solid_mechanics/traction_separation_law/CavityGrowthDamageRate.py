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

"""Cavitated-area-fraction rate contributed by cavity volume growth."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sqrt
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("CavityGrowthDamageRate")
class CavityGrowthDamageRate(Model):
    r"""The part of $\dot f$ driven by growth of the existing cavities,

    $$
    \dot{f}_\text{growth}
      = \frac{\sqrt{\pi}\, N^{3/2}}{2 h \sqrt{f}} \dot{V}.
    $$

    This is the geometric conversion from a single-cavity volume rate to an
    area-fraction rate, obtained from $f = \pi a^2 N$ and
    $V = \tfrac{4}{3}\pi a^3 h$ by eliminating $a$. Feeding it the diffusive
    rate $\dot V = 8\pi\mathcal{D}\langle T_n\rangle / q(f^*)$ recovers the
    Sham-Needleman growth term
    $4\pi^{3/2}\mathcal{D} N^{3/2} \langle T_n \rangle / (h \sqrt{f} q(f^*))$
    exactly.

    Keeping the conversion separate from the growth law is what lets a second
    growth mechanism be summed onto $\dot V$ (with a
    `ScalarLinearCombination`) without restating this geometry.

    Note the *physical* $f$ appears here, not the effective $f^*$: the
    coalescence map accelerates the softening functions only, so that the
    identity $[\![u]\!]_n^v = NV - N_0V_0$ keeps holding.
    """

    hit = HitSchema(
        input("cavity_fraction", Scalar, "Physical cavitated area fraction"),
        input("cavity_density", Scalar, "Cavity number density per unit boundary area"),
        input("cavity_volume_rate", Scalar, "Rate of change of a single cavity volume"),
        output(
            "cavity_fraction_growth_rate",
            Scalar,
            "Area fraction rate contributed by growth of existing cavities",
        ),
        parameter(
            "cavity_shape_factor",
            Scalar,
            "Cavity shape factor h(psi); h is about 0.61 at a tip angle of 75 degrees",
            attr="h",
            allow_promotion=True,
        ),
    )

    h: Scalar

    def forward(  # type: ignore[override]
        self,
        f: Scalar,
        N: Scalar,
        Vdot: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        h = self._get_param("h", promoted_params, Scalar)

        root_pi = math.sqrt(math.pi)
        sqrt_f = sqrt(f)
        sqrt_N = sqrt(N)
        # geo = sqrt(pi) * N^{3/2} / (2 h sqrt(f))
        geo = root_pi * N * sqrt_N / (2.0 * h * sqrt_f)
        fdot = geo * Vdot

        if v is None:
            return fdot

        dfdot_df = -0.5 * geo * Vdot / f
        dfdot_dN = 1.5 * geo * Vdot / N
        dfdot_dVdot = geo

        actions: dict[str, ChainRuleAction] = {
            "cavity_fraction": lambda V, c=dfdot_df: c * V,
            "cavity_density": lambda V, c=dfdot_dN: c * V,
            "cavity_volume_rate": lambda V, c=dfdot_dVdot: c * V,
        }
        if "h" in self._promoted_params:
            dfdot_dh = -fdot / h
            actions[self._promoted_params["h"].input_name] = lambda V, c=dfdot_dh: c * V
        return fdot, self.apply_chain_rule(v, "cavity_fraction_growth_rate", actions, output=fdot)


__all__ = ["CavityGrowthDamageRate"]
