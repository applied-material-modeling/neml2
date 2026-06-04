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

"""Python-native mirror of the C++ ``KocksMeckingFlowViscosity`` model."""

from __future__ import annotations

import math

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import BLOCK_NAME, HitSchema, input, option, output, parameter
from ....types import Scalar, exp


@register_neml2_object("KocksMeckingFlowViscosity")
class KocksMeckingFlowViscosity(Model):
    r"""Calculates the temperature-dependent flow viscosity for a Perzyna-type
    model using the Kocks-Mecking model.  The value is
    $\eta = \exp{B} \mu \dot{\varepsilon}_0^\frac{-k T A}{\mu b^3}$ with
    $\mu$ the shear modulus, $\dot{\varepsilon}_0$ a reference
    strain rate,  $b$ the Burgers vector, $k$ the Boltzmann
    constant, $T$ absolute temperature, $A$ the Kocks-Mecking slope
    parameter, and $B$ the Kocks-Mecking intercept parameter.
    """

    hit = HitSchema(
        input("temperature", Scalar, "Absolute temperature"),
        # The C++ model registers its output as ``declare_output_variable<Scalar>(name())``
        # -- the HIT block name -- with no user-facing option. Mirror that with
        # ``default=BLOCK_NAME`` so silent HIT resolves to the block name.
        output(
            "viscosity",
            Scalar,
            "Output name of the flow viscosity",
            default=BLOCK_NAME,
            attr="_viscosity",
        ),
        parameter(
            "A",
            Scalar,
            "The Kocks-Mecking slope parameter",
            attr="A",
            allow_nonlinear=True,
        ),
        parameter(
            "B",
            Scalar,
            "The Kocks-Mecking intercept parameter",
            attr="B",
            allow_nonlinear=True,
        ),
        parameter(
            "shear_modulus",
            Scalar,
            "The shear modulus",
            attr="mu",
            allow_nonlinear=True,
        ),
        option("eps0", float, "The reference strain rate", attr="_eps0"),
        option("k", float, "Boltzmann constant", attr="_k"),
        option("b", float, "The Burgers vector", attr="_b"),
    )

    # ``from_hit`` auto-declares the three parameters (stored as ``A``, ``B``,
    # ``mu``); the three float options land on ``self`` via ``attr=``. Annotate
    # so pyright sees the typed wrappers that ``Model.__getattr__`` returns
    # rather than ``nn.Module``'s generic ``Module`` hint.
    A: Scalar
    B: Scalar
    mu: Scalar
    _eps0: float
    _k: float
    _b: float
    _b3: float
    _log_eps0: float
    _viscosity: str

    def __post_init__(self) -> None:
        # Cache b^3 and log(eps0); the C++ constructor stores ``_b3 = b*b*b``
        # and the set_value body repeatedly calls ``std::log(_eps0)``.
        self._b3 = self._b * self._b * self._b
        self._log_eps0 = math.log(self._eps0)

    def forward(  # type: ignore[override]
        self,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``KocksMeckingFlowViscosity::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingFlowViscosity.cxx``.
        # Note: the docstring formula's exponent is written with a leading
        # minus sign, but the C++ source uses ``pow(eps0, k*T*A/(mu*b^3))``
        # (positive). The implementation tracks the source; for typical
        # ``A < 0`` the exponent is naturally negative.
        T = temperature
        A = self._get_param("A", nl_params, Scalar)
        B = self._get_param("B", nl_params, Scalar)
        mu = self._get_param("mu", nl_params, Scalar)

        # post = eps0^(k T A / (mu b^3)) = exp((k T A / (mu b^3)) * log(eps0))
        c = (self._k * T * A) / (mu * self._b3)
        post = exp(c * self._log_eps0)
        expB = exp(B)
        eta = expB * mu * post

        if v is None:
            return eta

        # Differential pushforward. Each action takes a Scalar tangent
        # V and returns the Scalar tangent of eta. Closed-form coefficients
        # mirror the dense C++ Jacobian in set_value:
        #   d eta / d T   =  A * exp(B) * k * log(eps0) * post / b^3
        #   d eta / d A   =  exp(B) * k * T * log(eps0) * post / b^3
        #   d eta / d B   =  exp(B) * mu * post = eta
        #   d eta / d mu  =  exp(B) * post * (1 - A*k*T*log(eps0) / (b^3 * mu))
        deta_dT = A * expB * self._k * self._log_eps0 * post / self._b3

        actions: dict[str, ChainRuleAction] = {
            "temperature": lambda V, c=deta_dT: c * V,
        }
        if "A" in self._nl_params:
            deta_dA = expB * self._k * T * self._log_eps0 * post / self._b3
            actions[self._nl_params["A"].input_name] = lambda V, c=deta_dA: c * V
        if "B" in self._nl_params:
            actions[self._nl_params["B"].input_name] = lambda V, c=eta: c * V
        if "mu" in self._nl_params:
            deta_dmu = expB * post * (1.0 - A * self._k * T * self._log_eps0 / (self._b3 * mu))
            actions[self._nl_params["mu"].input_name] = lambda V, c=deta_dmu: c * V

        return eta, self.apply_chain_rule(v, self._viscosity, actions, output=eta)


__all__ = ["KocksMeckingFlowViscosity"]
