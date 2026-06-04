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

"""Python-native mirror of the C++ ``AssociativeKinematicPlasticHardening`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar


@register_neml2_object("AssociativeKinematicPlasticHardening")
class AssociativeKinematicPlasticHardening(Model):
    r"""Map the flow rate (i.e., the consistency parameter in the KKT conditions)
    to the rate of internal variables. This object calculates the rate of
    kinematic plastic strain following associative flow rule, i.e.
    $\dot{\boldsymbol{K}}_p = - \dot{\gamma} \frac{\partial f}{\partial \boldsymbol{X}}$, where
    $\dot{\boldsymbol{K}}_p$ is the kinematic
    plastic strain, $\dot{\gamma}$ is the flow rate, $f$ is the
    yield function, and $\boldsymbol{X}$ is the kinematic hardening.
    """

    hit = HitSchema(
        input("flow_rate", Scalar, "Flow rate"),
        input(
            "kinematic_hardening_direction",
            SR2,
            "Direction of associative kinematic hardening which can be calculated using Normality.",
        ),
        output("kinematic_plastic_strain_rate", SR2, "Rate of kinematic plastic strain"),
    )

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        kinematic_hardening_direction: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Kp_dot = - gamma_dot * NX (linear in both inputs).
        Kp_dot = -(flow_rate * kinematic_hardening_direction)
        if v is None:
            return Kp_dot
        # Differential pushforward:
        #   d(Kp_dot)/d(flow_rate) acting on Scalar tangent V -> -NX * V (SR2)
        #   d(Kp_dot)/d(NX)        acting on SR2    tangent V -> -flow_rate * V (SR2)
        neg_NX = -kinematic_hardening_direction
        neg_flow = -flow_rate
        return Kp_dot, self.apply_chain_rule(
            v,
            "kinematic_plastic_strain_rate",
            {
                "flow_rate": lambda V, c=neg_NX: c * V,
                "kinematic_hardening_direction": lambda V, c=neg_flow: c * V,
            },
            output=Kp_dot,
        )
