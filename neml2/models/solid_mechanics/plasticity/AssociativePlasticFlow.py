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

"""Python-native mirror of the C++ ``AssociativePlasticFlow`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2, Scalar


@register_native("AssociativePlasticFlow")
class AssociativePlasticFlow(Model):
    r"""Map the flow rate (i.e., the consistency parameter in the KKT conditions)
    to the rate of internal variables. This object calculates the rate of
    plastic strain following associative flow rule, i.e.
    $\dot{\boldsymbol{\varepsilon}}_p = - \dot{\gamma} \frac{\partial f}{\partial \boldsymbol{M}}$,
    where
    $\dot{\boldsymbol{\varepsilon}}_p$ is the plastic strain,
    $\dot{\gamma}$ is the flow rate, $f$ is the yield function, and
    $\boldsymbol{M}$ is the Mandel stress.
    """

    hit = HitSchema(
        input("flow_rate", Scalar, "Flow rate"),
        input("flow_direction", SR2, "Flow direction which can be calculated using Normality"),
        output("plastic_strain_rate", SR2, "Rate of plastic strain"),
    )

    def forward(  # type: ignore[override]
        self,
        flow_rate: Scalar,
        flow_direction: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        ep_dot = flow_rate * flow_direction
        if v is None:
            return ep_dot
        # ∂ep_dot/∂flow_rate = flow_direction (SR2)·V_scalar;
        # ∂ep_dot/∂flow_direction = flow_rate (Scalar)·V_sr2.
        return ep_dot, self.apply_chain_rule(
            v,
            "plastic_strain_rate",
            {
                "flow_rate": lambda V, c=flow_direction: c * V,
                "flow_direction": lambda V, c=flow_rate: c * V,
            },
            output=ep_dot,
        )
