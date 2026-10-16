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

"""Smeared cavity volume expressed as a normal opening jump."""

from __future__ import annotations

import math

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, sqrt
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("CavityVolumeOpening")
class CavityVolumeOpening(Model):
    r"""The viscous (cavitation) part of the normal jump, which is the smeared
    cavity volume measured from the initial state,

    $$
    [\![u]\!]_n^v = N V(f, N) - N_0 V_0, \qquad
    N V(f, N) = \frac{4h}{3\sqrt{\pi}} \frac{f^{3/2}}{N^{1/2}}.
    $$

    $N_0$ and $f_0$ are the initial cavity density and area fraction, so a
    pristine boundary opens by zero. Both must be strictly positive: $f_0 = 0$
    is the singular point of the damage law.

    Differentiating this expression in time reproduces the Sham-Needleman
    viscous opening rate term by term, so the algebraic form is carried here
    rather than a rate; the time integration of $f$ and $N$ then supplies the
    rate implicitly and the identity holds exactly at every converged step
    instead of only to integration order.
    """

    hit = HitSchema(
        input("cavity_fraction", Scalar, "Physical cavitated area fraction"),
        input("cavity_density", Scalar, "Cavity number density per unit boundary area"),
        output(
            "viscous_normal_opening",
            Scalar,
            "Viscous part of the normal jump, i.e. the smeared cavity volume",
        ),
        parameter(
            "cavity_shape_factor",
            Scalar,
            "Cavity shape factor h(psi); h is about 0.61 at a tip angle of 75 degrees",
            attr="h",
            allow_promotion=True,
        ),
        parameter(
            "initial_cavity_fraction",
            Scalar,
            "Initial cavitated area fraction f_0, strictly positive",
            attr="f0",
            allow_promotion=True,
        ),
        parameter(
            "initial_cavity_density",
            Scalar,
            "Initial cavity number density N_0, strictly positive",
            attr="N0",
            allow_promotion=True,
        ),
    )

    h: Scalar
    f0: Scalar
    N0: Scalar

    def forward(  # type: ignore[override]
        self,
        f: Scalar,
        N: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        h = self._get_param("h", promoted_params, Scalar)
        f0 = self._get_param("f0", promoted_params, Scalar)
        N0 = self._get_param("N0", promoted_params, Scalar)

        c = 4.0 * h / (3.0 * math.sqrt(math.pi))
        NV = c * f * sqrt(f) / sqrt(N)
        NV0 = c * f0 * sqrt(f0) / sqrt(N0)
        opening = NV - NV0

        if v is None:
            return opening

        du_df = 1.5 * NV / f
        du_dN = -0.5 * NV / N

        actions: dict[str, ChainRuleAction] = {
            "cavity_fraction": lambda V, k=du_df: k * V,
            "cavity_density": lambda V, k=du_dN: k * V,
        }
        if "h" in self._promoted_params:
            du_dh = opening / h
            actions[self._promoted_params["h"].input_name] = lambda V, k=du_dh: k * V
        return opening, self.apply_chain_rule(v, "viscous_normal_opening", actions, output=opening)


__all__ = ["CavityVolumeOpening"]
