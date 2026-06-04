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

"""Python-native mirror of the C++ ``KocksMeckingIntercept`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict, SecondOrderChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, output, parameter
from ....types import Scalar


@register_neml2_object("KocksMeckingIntercept")
class KocksMeckingIntercept(Model):
    r"""The critical value of the normalized activation energy given by
    $g_0 \frac{C-B}{A}$
    """

    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        output("intercept", Scalar, "The intercept"),
        parameter("A", Scalar, "The Kocks-Mecking slope", attr="A", allow_nonlinear=True),
        parameter("B", Scalar, "The Kocks-Mecking intercept", attr="B", allow_nonlinear=True),
        parameter(
            "C", Scalar, "The Kocks-Mecking horizontal value", attr="C", allow_nonlinear=True
        ),
    )

    # ``from_hit`` auto-declares the A/B/C parameters via
    # ``declare_typed_parameter``; ``allow_nonlinear=True`` lets each
    # independently resolve through mode 1/2/3/4 (literal, [Tensors] ref,
    # [Models] wiring, or bare input promotion). Annotate so pyright sees
    # the typed wrapper that ``Model.__getattr__`` returns rather than
    # ``nn.Module``'s generic ``Module`` hint.
    A: Scalar
    B: Scalar
    C: Scalar

    def forward(  # type: ignore[override]
        self,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Mirrors ``KocksMeckingIntercept::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingIntercept.cxx``:
        # ``b = (C - B) / A``.
        A = self._get_param("A", nl_params, Scalar)
        B = self._get_param("B", nl_params, Scalar)
        C = self._get_param("C", nl_params, Scalar)

        diff = C - B
        inv_A = 1.0 / A
        b = diff * inv_A
        if v is None:
            return b

        # Differential pushforward. Closed-form first-order
        # coefficients mirror the dense C++ Jacobian in set_value:
        #   db/dA = -(C - B) / A^2
        #   db/dB = -1 / A
        #   db/dC =  1 / A
        # The forward has no structural input variables; every action key is
        # the nl-promoted input name a mode-3/4 parameter resolution produced.
        inv_A2 = inv_A * inv_A
        actions_1: dict = {}
        if "A" in self._nl_params:
            coef_A = -diff * inv_A2
            actions_1[self._nl_params["A"].input_name] = lambda V, c=coef_A: c * V
        if "B" in self._nl_params:
            coef_B = -inv_A
            actions_1[self._nl_params["B"].input_name] = lambda V, c=coef_B: c * V
        if "C" in self._nl_params:
            coef_C = inv_A
            actions_1[self._nl_params["C"].input_name] = lambda V, c=coef_C: c * V

        if v2 is None and vh is None:
            return b, *self.propagate_tangents(v, "intercept", actions_1, output=b)

        # Second-order pushforward. each lambda
        # receives primal-shape tangents and returns the primal-shape
        # bilinear ``coef * Va * Vb``; the framework's ``_apply_action_2``
        # handles the (N_a, N_b) seed-pair iteration + stacking. Bodies are
        # pure typed-wrapper algebra. The Hessian entries are:
        #   d2b/dA2     =  2 (C - B) / A^3
        #   d2b/dA dB   =  1 / A^2  (and the symmetric dB dA)
        #   d2b/dA dC   = -1 / A^2  (and the symmetric dC dA)
        #   d2b/dB2 = d2b/dC2 = d2b/dB dC = 0 (omitted)
        actions_2: dict = {}
        inv_A3 = inv_A2 * inv_A
        if "A" in self._nl_params:
            aname = self._nl_params["A"].input_name
            c_AA = 2.0 * diff * inv_A3
            actions_2[(aname, aname)] = lambda Va, Vb, c=c_AA: c * Va * Vb
            if "B" in self._nl_params:
                bname = self._nl_params["B"].input_name
                c_AB = inv_A2
                actions_2[(aname, bname)] = lambda Va, Vb, c=c_AB: c * Va * Vb
                actions_2[(bname, aname)] = lambda Va, Vb, c=c_AB: c * Va * Vb
            if "C" in self._nl_params:
                cname = self._nl_params["C"].input_name
                c_AC = -inv_A2
                actions_2[(aname, cname)] = lambda Va, Vb, c=c_AC: c * Va * Vb
                actions_2[(cname, aname)] = lambda Va, Vb, c=c_AC: c * Va * Vb

        return b, *self.propagate_tangents(
            v,
            "intercept",
            actions_1,
            output=b,
            v2=v2,
            actions_2=actions_2,
            vh=vh,
        )


__all__ = ["KocksMeckingIntercept"]
