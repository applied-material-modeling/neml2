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

"""Viscous grain boundary sliding rate, softened by cavitation."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("GrainBoundarySlidingRate")
class GrainBoundarySlidingRate(Model):
    r"""Linear-viscous grain boundary sliding, softened by the cavitated area
    fraction,

    $$
    \dot{[\![u]\!]}_t^v = \frac{T_t}{(1 - f^*)\, \eta}.
    $$

    $\eta$ is the grain boundary sliding viscosity and $(1 - f^*)$ is the
    uncavitated ligament fraction, so the boundary slides ever more freely as
    cavities coalesce. The factor vanishes at $f^* = 1$; keeping the effective
    fraction strictly below one (see
    :class:`~.EffectiveCavityFraction.EffectiveCavityFraction`) is what keeps
    this rate finite.

    This is the dashpot half of the sliding branch. The elastic half lives in
    :class:`~.ShamNeedlemanTraction.ShamNeedlemanTraction`, which places the
    two in series.
    """

    hit = HitSchema(
        input("shear_traction", Scalar, "Shear traction across the interface"),
        input(
            "effective_cavity_fraction",
            Scalar,
            "Effective cavitated area fraction softening the sliding viscosity",
        ),
        output("viscous_sliding_rate", Scalar, "Rate of the viscous part of the sliding jump"),
        parameter(
            "sliding_viscosity",
            Scalar,
            "Grain boundary sliding viscosity",
            attr="eta",
            allow_promotion=True,
        ),
    )

    eta: Scalar

    def forward(  # type: ignore[override]
        self,
        Tt: Scalar,
        f_star: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        eta = self._get_param("eta", promoted_params, Scalar)

        ligament = 1.0 - f_star
        denom = ligament * eta
        rate = Tt / denom

        if v is None:
            return rate

        dr_dTt = 1.0 / denom
        # d/df* of Tt / ((1 - f*) eta) = Tt / ((1 - f*)^2 eta)
        dr_dfs = Tt / (ligament * ligament * eta)

        actions: dict[str, ChainRuleAction] = {
            "shear_traction": lambda V, c=dr_dTt: c * V,
            "effective_cavity_fraction": lambda V, c=dr_dfs: c * V,
        }
        if "eta" in self._promoted_params:
            dr_deta = -rate / eta
            actions[self._promoted_params["eta"].input_name] = lambda V, c=dr_deta: c * V
        return rate, self.apply_chain_rule(v, "viscous_sliding_rate", actions, output=rate)


__all__ = ["GrainBoundarySlidingRate"]
