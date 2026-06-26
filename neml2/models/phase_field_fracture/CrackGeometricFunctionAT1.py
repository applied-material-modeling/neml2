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

"""Python-native mirror of the C++ ``CrackGeometricFunctionAT1`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar
from ..chain_rule import ChainRuleDict
from ..model import Model


@register_neml2_object("CrackGeometricFunctionAT1")
class CrackGeometricFunctionAT1(Model):
    r"""Crack geometric function associated with the AT-1 functional,
    $\alpha = d$ where $d$ is the phase-field variable.
    """

    hit = HitSchema(
        input("phase", Scalar, "Phase-field variable"),
        output("crack", Scalar, "Value of the crack geometric function"),
    )

    def forward(  # type: ignore[override]
        self,
        phase: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # AT-1 functional: alpha(d) = d. Identity map on the phase-field.
        alpha = phase
        if v is None:
            return alpha
        # Differential pushforward: d(alpha)/d(phase) = 1, so the
        # action on any incoming Scalar tangent V is just V itself.
        actions = {"phase": lambda V: V}
        return alpha, self.apply_chain_rule(v, "crack", actions, output=alpha)
