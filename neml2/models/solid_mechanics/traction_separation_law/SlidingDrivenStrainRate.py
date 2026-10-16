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

"""Local surrogate for the bulk strain rate that drives cavity nucleation."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sign
from ....types import abs as tensor_abs
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("SlidingDrivenStrainRate")
class SlidingDrivenStrainRate(Model):
    r"""Local stand-in for the bulk strain rate that drives nucleation,

    $$
    \dot{\varepsilon} = \frac{\left| \dot{[\![u]\!]}_t^v \right|}{d},
    $$

    with $d$ a characteristic grain or grain boundary length scale.

    The bulk strain rate is the only non-local quantity in the original
    Sham-Needleman model. Assuming nucleation is driven by sliding stress
    concentrations replaces it with this purely local expression, which is
    what makes the simplified model usable as an interface law without a
    coupling back to the surrounding continuum.

    This leaf is optional by construction: the nucleation law consumes
    ``strain_rate`` as an ordinary input, so an input file may instead supply
    the true bulk rate (for example through a `ScalarInputParameter`) and
    leave this model out of the composition entirely.
    """

    hit = HitSchema(
        input(
            "viscous_sliding_rate",
            Scalar,
            "Rate of the viscous part of the sliding jump",
        ),
        output("strain_rate", Scalar, "Effective strain rate driving cavity nucleation"),
        parameter(
            "length_scale",
            Scalar,
            "Characteristic grain or grain boundary length scale",
            attr="d",
            allow_promotion=True,
        ),
    )

    d: Scalar

    def forward(  # type: ignore[override]
        self,
        slip_rate: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        d = self._get_param("d", promoted_params, Scalar)

        rate = tensor_abs(slip_rate) / d

        if v is None:
            return rate

        # d|x|/dx = sign(x); sign(0) = 0 picks the subgradient at the kink,
        # which is the symmetric choice and keeps the pushforward bounded.
        dr_dslip = sign(slip_rate) / d

        actions: dict[str, ChainRuleAction] = {
            "viscous_sliding_rate": lambda V, c=dr_dslip: c * V,
        }
        if "d" in self._promoted_params:
            dr_dd = -rate / d
            actions[self._promoted_params["d"].input_name] = lambda V, c=dr_dd: c * V
        return rate, self.apply_chain_rule(v, "strain_rate", actions, output=rate)


__all__ = ["SlidingDrivenStrainRate"]
