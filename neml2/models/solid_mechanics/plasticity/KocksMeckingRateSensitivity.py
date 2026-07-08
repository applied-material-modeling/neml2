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

"""Python-native mirror of the C++ ``KocksMeckingRateSensitivity`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, option, output, parameter
from ....types import Scalar
from ...chain_rule import ChainRuleDict
from ...model import Model


@register_neml2_object("KocksMeckingRateSensitivity")
class KocksMeckingRateSensitivity(Model):
    r"""Calculates the temperature-dependent rate sensitivity for a Perzyna-type model using the
    Kocks-Mecking model.  The value is $n = \frac{\mu b^3}{k T A}$ with $\mu$ the
    shear modulus, $b$ the Burgers vector, $k$ the Boltzmann constant, $T$
    absolute temperature, and $A$ the Kocks-Mecking slope parameter.
    """

    hit = HitSchema(
        input("temperature", Scalar, "Absolute temperature"),
        output("rate_sensitivity", Scalar, "Output name of the rate sensitivity"),
        parameter("A", Scalar, "The Kocks-Mecking slope parameter", attr="A", allow_promotion=True),
        parameter(
            "shear_modulus",
            Scalar,
            "The shear modulus",
            attr="mu",
            allow_promotion=True,
        ),
        option("k", float, "Boltzmann constant", attr="_k"),
        option("b", float, "The Burgers vector", attr="_b"),
    )

    # ``from_hit`` auto-declares the ``A`` / ``shear_modulus`` parameters
    # (stored as ``A`` / ``mu``); the two float options land on ``self`` via
    # ``attr=``. Annotate so pyright sees the typed wrappers that
    # ``Model.__getattr__`` returns.
    A: Scalar
    mu: Scalar
    _k: float
    _b: float
    _b3: float

    def __post_init__(self) -> None:
        # Cache the Burgers-vector cube so the forward stays a single scalar
        # multiply, matching the C++ constructor that stores
        # ``_b3 = b*b*b``.
        self._b3 = self._b * self._b * self._b

    def forward(  # type: ignore[override]
        self,
        temperature: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``KocksMeckingRateSensitivity::set_value`` in
        # ``src/neml2/models/solid_mechanics/plasticity/KocksMeckingRateSensitivity.cxx``:
        # ``n = -mu * b^3 / (k * T * A)``.
        T = temperature
        A = self._get_param("A", promoted_params, Scalar)
        mu = self._get_param("mu", promoted_params, Scalar)

        m = -mu * self._b3 / (self._k * T * A)

        if v is None:
            return m

        # Differential pushforward. Each action takes a Scalar tangent
        # V and returns the Scalar tangent of m. Closed-form coefficients
        # mirror the dense C++ Jacobian in set_value:
        #   dm/dT  =  b^3 * mu / (A * k * T^2)
        #   dm/dmu = -b^3        / (A * k * T)
        #   dm/dA  =  b^3 * mu / (A^2 * k * T)
        dm_dT = self._b3 * mu / (A * self._k * T * T)

        actions = {"temperature": lambda V, c=dm_dT: c * V}
        if "mu" in self._promoted_params:
            dm_dmu = -Scalar.from_value(self._b3, like=T) / (A * self._k * T)
            actions[self._promoted_params["mu"].input_name] = lambda V, c=dm_dmu: c * V
        if "A" in self._promoted_params:
            dm_dA = self._b3 * mu / (A * A * self._k * T)
            actions[self._promoted_params["A"].input_name] = lambda V, c=dm_dA: c * V

        return m, self.apply_chain_rule(v, "rate_sensitivity", actions, output=m)


__all__ = ["KocksMeckingRateSensitivity"]
