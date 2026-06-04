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

"""Python-native mirror of the C++ ``PerzynaPlasticFlowRate`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, abs, heaviside, log
from ....types import pow as wpow


@register_neml2_object("PerzynaPlasticFlowRate")
class PerzynaPlasticFlowRate(Model):
    r"""Perzyna's viscous approximation of the consistent yield envelope (with a
    power law), i.e. $\dot{\gamma} = \left( \frac{\left< f \right>}{\eta} \right)^n$, where $f$ is
    the yield function, $\eta$ is the
    reference stress, and $n$ is the power-law exponent.
    """

    # Variable names match the C++ defaults in PlasticFlowRate (yield_function,
    # flow_rate); HIT renames cascade via the schema's option-name ==
    # canonical-key convention.
    hit = HitSchema(
        input("yield_function", Scalar, "Yield function"),
        output("flow_rate", Scalar, "Flow rate"),
        parameter(
            "reference_stress",
            Scalar,
            "Reference stress",
            attr="eta",
            allow_nonlinear=True,
        ),
        parameter(
            "exponent",
            Scalar,
            "Power-law exponent",
            attr="n",
            allow_nonlinear=True,
        ),
    )

    # ``from_hit`` auto-declares the two parameters (stored as ``eta`` and
    # ``n``) — no __init__ needed.
    eta: Scalar
    n: Scalar

    def forward(  # type: ignore[override]
        self,
        yield_function: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``PerzynaPlasticFlowRate::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/PerzynaPlasticFlowRate.cxx``.
        f = yield_function
        eta = self._get_param("eta", nl_params, Scalar)
        n = self._get_param("n", nl_params, Scalar)

        Hf = heaviside(f)
        f_abs = abs(f)
        gamma_dot_m = wpow(f_abs / eta, n)
        gamma_dot = gamma_dot_m * Hf

        if v is None:
            return gamma_dot

        # Differential pushforward. Each action takes a Scalar tangent
        # V and returns the Scalar tangent of gamma_dot. Derivations match the
        # dense C++ Jacobian in set_value:
        #   d gamma_dot / d f   = (n / f_abs) * gamma_dot
        #   d gamma_dot / d eta = -n * gamma_dot / eta
        #   d gamma_dot / d n   = gamma_dot * log(f_abs / eta)
        coef_f = n / f_abs * gamma_dot

        actions = {"yield_function": lambda V, c=coef_f: c * V}
        if "eta" in self._nl_params:
            actions[self._nl_params["eta"].input_name] = lambda V, c=-n * gamma_dot / eta: c * V
        if "n" in self._nl_params:
            n_coef = gamma_dot * log(f_abs / eta)
            actions[self._nl_params["n"].input_name] = lambda V, c=n_coef: c * V

        return gamma_dot, self.apply_chain_rule(v, "flow_rate", actions, output=gamma_dot)
