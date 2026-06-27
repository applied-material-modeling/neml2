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

"""Python-native mirror of the C++ ``KocksMeckingFlowSwitch`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, option, output, parameter
from ....types import Scalar, cosh, tanh
from ....types import pow as wpow
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("KocksMeckingFlowSwitch")
class KocksMeckingFlowSwitch(Model):
    """Switches between rate independent and rate dependent flow rules based on the value of the
    Kocks-Mecking normalized activation energy.  For activation energies less than the threshold
    use the rate independent flow rule, for values greater than the threshold use the rate
    dependent flow rule.  This version uses a soft switch between the models, based on a tanh
    sigmoid function.
    """

    hit = HitSchema(
        input("activation_energy", Scalar, "The input name of the activation energy"),
        input(
            "rate_independent_flow_rate",
            Scalar,
            "Input name of the rate independent flow rate",
        ),
        input(
            "rate_dependent_flow_rate",
            Scalar,
            "Input name of the rate dependent flow rate",
        ),
        output("flow_rate", Scalar, "Output name for the mixed flow rate"),
        parameter(
            "g0",
            Scalar,
            "Critical value of activation energy",
            attr="g0",
            allow_promotion=True,
        ),
        option(
            "sharpness",
            float,
            (
                "A steepness parameter that controls the tanh mixing of the models.  Higher "
                "values gives a sharper transition."
            ),
            default=1.0,
            attr="sharp",
        ),
    )

    # ``from_hit`` auto-declares the ``g0`` parameter and stores ``sharpness``
    # under ``self.sharp`` via the option ``attr``; no __init__ needed.
    g0: Scalar
    sharp: float

    def forward(  # type: ignore[override]
        self,
        activation_energy: Scalar,
        rate_independent_flow_rate: Scalar,
        rate_dependent_flow_rate: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``KocksMeckingFlowSwitch::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingFlowSwitch.cxx``:
        #   sig       = (tanh(sharp * (g - g0)) + 1) / 2
        #   flow_rate = sig * rd_flow + (1 - sig) * ri_flow
        g = activation_energy
        ri = rate_independent_flow_rate
        rd = rate_dependent_flow_rate
        g0 = self._get_param("g0", promoted_params, Scalar)
        sharp = self.sharp

        arg = sharp * (g - g0)
        sig = (tanh(arg) + 1.0) / 2.0
        one_minus_sig = 1.0 - sig
        gamma_dot = sig * rd + one_minus_sig * ri

        if v is None:
            return gamma_dot

        # Differential pushforward. Coefficients precomputed once;
        # each action is just a Scalar*Scalar multiplication, no Jacobian.
        # d gamma_dot / d rd_flow = sig
        # d gamma_dot / d ri_flow = 1 - sig
        # d gamma_dot / d g       = 0.5 * sharp * sech^2(sharp*(g-g0)) * (rd - ri)
        # d gamma_dot / d g0      = -(d gamma_dot / d g)
        sech2 = wpow(cosh(arg), -2.0)
        d_dg = 0.5 * sharp * sech2 * (rd - ri)

        actions = {
            "activation_energy": lambda V, c=d_dg: c * V,
            "rate_independent_flow_rate": lambda V, c=one_minus_sig: c * V,
            "rate_dependent_flow_rate": lambda V, c=sig: c * V,
        }
        if "g0" in self._promoted_params:
            actions[self._promoted_params["g0"].input_name] = lambda V, c=-d_dg: c * V

        return gamma_dot, self.apply_chain_rule(v, "flow_rate", actions, output=gamma_dot)
