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

"""Python-native mirror of the C++ ``AssociativeJ2FlowDirection`` model."""

from __future__ import annotations

import math

import torch

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, output
from ....types import SR2
from ....types.functions import dev, inner, norm


@register_native("AssociativeJ2FlowDirection")
class AssociativeJ2FlowDirection(Model):
    """The plastic flow direction assuming an associative J2 flow."""

    hit = HitSchema(
        input("mandel_stress", SR2, "Mandel stress"),
        output("flow_direction", SR2, "Flow direction"),
    )

    # Mirrors C++ ``machine_precision`` regularizer used inside ``norm``.
    _SQRT_3_2 = math.sqrt(3.0 / 2.0)

    def forward(  # type: ignore[override]
        self,
        mandel_stress: SR2,
        v: ChainRuleDict | None = None,
    ) -> SR2 | tuple[SR2, ChainRuleDict]:
        # Match the C++ ``set_value`` regularization eps = machine_precision(dtype).
        eps = torch.finfo(mandel_stress.dtype).eps
        S = dev(mandel_stress)
        vm = self._SQRT_3_2 * norm(S, eps)
        # N = (3/2) * S / vm  (the associative J2 flow direction).
        N = (1.5 / vm) * S

        if v is None:
            return N

        # Differential of N w.r.t. M, written as pure wrapper algebra
        # (D-062 -- no SSR4 materialisation, no .data, no torch.<op>).
        #
        # Let S = dev(M), vm = sqrt(3/2) ||S||, N = (3/2) S / vm.
        # Then dS = dev(V), dvm = (3/(2 vm)) (S : dS) = N : dS, and
        #     dN = (3/(2 vm)) dS - (1/vm) N (N : dS).
        inv_vm = 1.0 / vm

        def mandel_stress_action(V: SR2) -> SR2:
            dS = dev(V)
            return 1.5 * inv_vm * dS - inv_vm * inner(N, dS) * N

        return N, self.apply_chain_rule(
            v,
            "flow_direction",
            {"mandel_stress": mandel_stress_action},
            output=N,
        )
