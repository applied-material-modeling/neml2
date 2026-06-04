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

"""Python-native mirror of the C++ ``KocksMeckingYieldStress`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleAction, ChainRuleDict, SecondOrderChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, output, parameter
from ....types import Scalar, exp


@register_neml2_object("KocksMeckingYieldStress")
class KocksMeckingYieldStress(Model):
    r"""The yield stress given by the Kocks-Mecking model. $\sigma_y = \exp{C} \mu$ with $\mu$ the
    shear modulus and $C$ the
    horizontal intercept from the Kocks-Mecking diagram.
    """

    # Smooth closed-form (product of mu and exp(C)) ⇒ second-order chain rule
    # is well-defined. Required so the wrapping Normality can reuse v2/vh
    # through this leaf.
    SUPPORTS_SECOND_ORDER = True

    hit = HitSchema(
        output("yield_stress", Scalar, "Output name of the yield stress"),
        parameter(
            "C",
            Scalar,
            "The Kocks-Mecking horizontal intercept",
            attr="C",
            allow_nonlinear=True,
        ),
        parameter(
            "shear_modulus",
            Scalar,
            "The shear modulus",
            attr="mu",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the ``C`` / ``shear_modulus`` parameters
    # (stored as ``C`` / ``mu``) -- no __init__ needed.
    C: Scalar
    mu: Scalar

    def forward(  # type: ignore[override]
        self,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
        v2: SecondOrderChainRuleDict | None = None,
        vh: ChainRuleDict | None = None,
    ):
        # Mirrors ``KocksMeckingYieldStress::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingYieldStress.cxx``.
        C = self._get_param("C", nl_params, Scalar)
        mu = self._get_param("mu", nl_params, Scalar)

        # tau = mu * exp(C)
        eC = exp(C)
        tau = mu * eC
        if v is None:
            return tau

        # Differential pushforward. The leaf has no structural inputs;
        # only the two parameters can be promoted to runtime inputs. Coefficients
        # (Scalars) scale the incoming leading-K tangent automatically.
        #
        # Derivation (matches the dense C++ Jacobian in set_value):
        #   dtau / d mu = exp(C)
        #   dtau / d C  = mu * exp(C) = tau
        actions_1: dict[str, ChainRuleAction] = {}
        if "mu" in self._nl_params:
            actions_1[self._nl_params["mu"].input_name] = lambda V, c=eC: c * V
        if "C" in self._nl_params:
            actions_1[self._nl_params["C"].input_name] = lambda V, c=tau: c * V

        if v2 is None and vh is None:
            return tau, self.apply_chain_rule(v, "yield_stress", actions_1, output=tau)

        # Second-order pushforward. The C++
        # source pins three non-zero Hessian entries:
        #   d2tau/dC dC   = mu * exp(C) = tau
        #   d2tau/dC dmu  = exp(C) = eC      (and the symmetric dmu dC)
        #   d2tau/dmu dmu = 0                (omitted)
        # Each ``action_2(Va, Vb)`` receives primal-shape tangent slices
        # (NO leading seed dim) and returns a primal-shape Scalar bilinear;
        # the framework iterates the (N_a, N_b) seed-pair outer.
        actions_2: dict = {}
        if "C" in self._nl_params:
            cname = self._nl_params["C"].input_name
            actions_2[(cname, cname)] = lambda Va, Vb, c=tau: c * Va * Vb
            if "mu" in self._nl_params:
                mname = self._nl_params["mu"].input_name
                actions_2[(cname, mname)] = lambda Va, Vb, c=eC: c * Va * Vb
                actions_2[(mname, cname)] = lambda Va, Vb, c=eC: c * Va * Vb

        return tau, *self.propagate_tangents(
            v, "yield_stress", actions_1, output=tau, v2=v2, actions_2=actions_2, vh=vh
        )
