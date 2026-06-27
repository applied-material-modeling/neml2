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

"""Python-native mirror of the C++ ``CrackGeometricFunctionAT2`` model."""

from __future__ import annotations

from ...factory import register_neml2_object
from ...schema import HitSchema, input, output
from ...types import Scalar
from ..chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ..model import Model


@register_neml2_object("CrackGeometricFunctionAT2")
class CrackGeometricFunctionAT2(Model):
    r"""Crack geometric function associated with the AT-2 functional,
    $\alpha = d^2$ where $d$ is the phase-field variable.
    """

    # Smooth in d ⇒ second-order chain rule is well-defined. Required for any
    # ``Normality(energy)`` wrap that consumes this leaf inside the inner
    # composition (e.g. the elastic-brittle-fracture regression scenario).
    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        input("phase", Scalar, "Phase-field variable"),
        output("crack", Scalar, "Value of the crack geometric function"),
    )

    def forward(  # type: ignore[override]
        self,
        phase: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # AT-2 functional: alpha(d) = d**2.
        alpha = phase * phase
        if v is None:
            return alpha

        # First-order pushforward: d(alpha)/d(phase) = 2*d.
        def phase_action(V: Scalar) -> Scalar:
            return 2.0 * phase * V

        actions_1 = {"phase": phase_action}

        # Second-order Hessian: d^2(alpha)/d(phase)^2 = 2 (constant). The
        # framework iterates (k_a, k_b) seed pairs and stacks; this body is
        # pure typed-wrapper algebra — no .data, no outer.
        def phase_phase_action(Va: Scalar, Vb: Scalar) -> Scalar:
            return 2.0 * Va * Vb

        actions_2 = {("phase", "phase"): phase_phase_action}
        return alpha, *self.propagate_tangents(
            v, "crack", actions_1, output=alpha, v2=v2, actions_2=actions_2, vh=vh
        )
