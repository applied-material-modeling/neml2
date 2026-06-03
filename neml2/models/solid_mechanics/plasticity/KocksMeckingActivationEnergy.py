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

"""Python-native mirror of the C++ ``KocksMeckingActivationEnergy`` model."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_native
from ....model import Model
from ....schema import HitSchema, input, option, output, parameter
from ....types import Scalar, log


@register_native("KocksMeckingActivationEnergy")
class KocksMeckingActivationEnergy(Model):
    r"""Calculates the Kocks-Mecking normalized activation as
    $g = \frac{kT}{\mu b^3} \log \frac{\dot{\varepsilon}_0}{\dot{\varepsilon}}$
    with $\mu$ the shear modulus, $k$ the Boltzmann constant,
    $T$ the absolute temperature, $b$ the Burgers vector length,
    $\dot{\varepsilon}_0$ a reference strain rate, and
    $\dot{\varepsilon}$ the current strain rate.
    """

    hit = HitSchema(
        input("temperature", Scalar, "Absolute temperature"),
        input("strain_rate", Scalar, "Effective strain rate"),
        output("activation_energy", Scalar, "Output name of the activation energy"),
        parameter(
            "shear_modulus",
            Scalar,
            "The shear modulus",
            attr="mu",
            allow_nonlinear=True,
        ),
        option("eps0", float, "Reference strain rate", attr="_eps0"),
        option("k", float, "The Boltzmann constant", attr="_k"),
        option("b", float, "Magnitude of the Burgers vector", attr="_b"),
    )

    # ``from_hit`` auto-declares the ``shear_modulus`` parameter (stored as
    # ``mu``); the three float options land on ``self`` via ``attr=``. Annotate
    # so pyright sees the typed wrappers that ``Model.__getattr__`` returns
    # rather than ``nn.Module``'s generic ``Module`` hint.
    mu: Scalar
    _eps0: float
    _k: float
    _b: float
    _b3: float

    def __post_init__(self) -> None:
        # Cache the Burgers-vector cube so the forward stays a single scalar
        # multiply, matching the C++ constructor that stores ``_b3 = b*b*b``.
        self._b3 = self._b * self._b * self._b

    def forward(  # type: ignore[override]
        self,
        temperature: Scalar,
        strain_rate: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``KocksMeckingActivationEnergy::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingActivationEnergy.cxx``:
        # ``g = (k * T) / (mu * b^3) * log(eps0 / eps_dot)``.
        T = temperature
        eps_dot = strain_rate
        mu = self._get_param("mu", nl_params, Scalar)

        log_ratio = log(Scalar.from_value(self._eps0, like=eps_dot) / eps_dot)
        denom = mu * self._b3
        g = (self._k * T) / denom * log_ratio

        if v is None:
            return g

        # Differential pushforward. Each action takes a Scalar tangent
        # V and returns the Scalar tangent of g. Closed-form coefficients
        # mirror the dense C++ Jacobian in set_value:
        #   dg/dT       =  k / (mu * b^3) * log(eps0 / eps_dot)
        #   dg/d eps    = -k * T / (mu * b^3 * eps_dot)
        #   dg/d mu     = -k * T / (b^3 * mu^2) * log(eps0 / eps_dot)
        dg_dT = self._k / denom * log_ratio
        dg_deps = -self._k * T / (denom * eps_dot)

        actions = {
            "temperature": lambda V, c=dg_dT: c * V,
            "strain_rate": lambda V, c=dg_deps: c * V,
        }
        if "mu" in self._nl_params:
            dg_dmu = -self._k * T / (self._b3 * mu * mu) * log_ratio
            actions[self._nl_params["mu"].input_name] = lambda V, c=dg_dmu: c * V

        return g, self.apply_chain_rule(v, "activation_energy", actions, output=g)


__all__ = ["KocksMeckingActivationEnergy"]
