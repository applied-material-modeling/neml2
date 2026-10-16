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

"""Diffusive (Hull-Rimmer) single-cavity volume growth rate."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, log, macaulay, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


def cavity_softening(f_star: Scalar) -> tuple[Scalar, Scalar]:
    r"""The softening function $q(f^*) = 2\ln(1/f^*) - (3 - f^*)(1 - f^*)$ and
    its derivative $q'(f^*) = -2/f^* + 4 - 2 f^*$.

    Returned together because every consumer that divides by $q$ also needs
    $q'$ for its pushforward, and re-deriving it at each call site is how the
    two drift apart.

    $q$ vanishes at $f^* = 1$, so callers must keep $f^*$ strictly below one
    (that is what the ``ultimate_fraction`` knob on
    :class:`~.EffectiveCavityFraction.EffectiveCavityFraction` is for).
    """
    q = -2.0 * log(f_star) - (3.0 - f_star) * (1.0 - f_star)
    dq = -2.0 / f_star + 4.0 - 2.0 * f_star
    return q, dq


@register_neml2_object("DiffusiveCavityGrowthRate")
class DiffusiveCavityGrowthRate(Model):
    r"""Grain-boundary-diffusion-controlled growth of a single cavity,

    $$
    \dot{V} = \frac{8 \pi \mathcal{D} \langle T_n \rangle}{q(f^*)}, \qquad
    q(f) = 2\ln\frac{1}{f} - (3-f)(1-f),
    $$

    with $\mathcal{D} = D_B \delta_B \Omega / (k \theta)$ the lumped grain
    boundary diffusion parameter. The $8\pi$ prefactor is tied to this form of
    $q$.

    This is the only cavity growth mechanism in the simplified Sham-Needleman
    model: creep-assisted growth and the Rice constraint are both omitted. It
    is a separate leaf precisely so further growth terms can be summed onto it
    (via a `ScalarLinearCombination`) without touching the damage or opening
    laws that consume $\dot{V}$.

    The Macaulay bracket on $T_n$ disables sintering under compression, which
    also keeps $f$ away from the $f \to 0$ singularity in the damage law.
    """

    hit = HitSchema(
        input("normal_traction", Scalar, "Normal traction across the interface"),
        input(
            "effective_cavity_fraction",
            Scalar,
            "Effective cavitated area fraction driving the softening function q",
        ),
        output("cavity_volume_rate", Scalar, "Rate of change of a single cavity volume"),
        parameter(
            "diffusivity",
            Scalar,
            "Lumped grain boundary diffusion parameter D = D_B delta_B Omega / (k theta)",
            attr="D",
            allow_promotion=True,
        ),
    )

    D: Scalar

    def forward(  # type: ignore[override]
        self,
        Tn: Scalar,
        f_star: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        D = self._get_param("D", promoted_params, Scalar)

        # Detached tension mask: the Macaulay kink is not differentiable, and
        # the compression branch must contribute nothing through Tn.
        tensile = gt(Tn, 0.0)
        Tn_pos = macaulay(Tn)
        q, dq = cavity_softening(f_star)

        pref = 8.0 * math.pi * D
        Vdot = pref * Tn_pos / q

        if v is None:
            return Vdot

        zero = Scalar.from_value(0.0, like=Tn)
        dV_dTn = where(tensile, pref / q, zero)
        dV_dfs = -pref * Tn_pos * dq / (q * q)

        actions: dict[str, ChainRuleAction] = {
            "normal_traction": lambda V, c=dV_dTn: c * V,
            "effective_cavity_fraction": lambda V, c=dV_dfs: c * V,
        }
        if "D" in self._promoted_params:
            dV_dD = 8.0 * math.pi * Tn_pos / q
            actions[self._promoted_params["D"].input_name] = lambda V, c=dV_dD: c * V
        return Vdot, self.apply_chain_rule(v, "cavity_volume_rate", actions, output=Vdot)


__all__ = ["DiffusiveCavityGrowthRate", "cavity_softening"]
