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

"""Stress-driven continuous cavity nucleation."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, pow, where
from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...model import Model


@register_neml2_object("CavityNucleationRate")
class CavityNucleationRate(Model):
    r"""Continuous, stress-driven nucleation of grain boundary cavities,

    $$
    \dot{N} = F_N \left\langle \frac{T_n}{\Sigma_0} \right\rangle^{\beta}
              \dot{\varepsilon},
    $$

    with $F_N$ the nucleation coefficient, $\Sigma_0$ a reference stress and
    $\beta$ the stress exponent. The Macaulay bracket switches nucleation off
    under compression.

    $\dot{\varepsilon}$ is an ordinary input, so it may be fulfilled either by
    :class:`~.SlidingDrivenStrainRate.SlidingDrivenStrainRate` (the local
    surrogate) or by a true bulk strain rate supplied from the input file.

    Leaving this model out of a composition and wiring the damage law without
    a ``cavity_density_rate`` gives the same model without nucleation, with
    $N$ held at its initial value.
    """

    hit = HitSchema(
        input("normal_traction", Scalar, "Normal traction across the interface"),
        input("strain_rate", Scalar, "Strain rate driving nucleation"),
        output("cavity_density_rate", Scalar, "Rate of change of the cavity number density"),
        parameter(
            "nucleation_coefficient",
            Scalar,
            "Nucleation coefficient F_N",
            attr="F_N",
            allow_promotion=True,
        ),
        parameter(
            "reference_stress",
            Scalar,
            "Nucleation reference stress Sigma_0",
            attr="S0",
            allow_promotion=True,
        ),
        parameter(
            "exponent",
            Scalar,
            "Nucleation stress exponent beta",
            attr="beta",
            allow_promotion=True,
        ),
    )

    F_N: Scalar
    S0: Scalar
    beta: Scalar

    def forward(  # type: ignore[override]
        self,
        Tn: Scalar,
        edot: Scalar,
        *promoted_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        F_N = self._get_param("F_N", promoted_params, Scalar)
        S0 = self._get_param("S0", promoted_params, Scalar)
        beta = self._get_param("beta", promoted_params, Scalar)

        # Detached tension mask plus a safe base: for beta < 1 the derivative
        # x^(beta-1) blows up at x = 0, so the compression branch must be
        # masked off *before* the power is taken, not after.
        tensile = gt(Tn, 0.0)
        zero = Scalar.from_value(0.0, like=Tn)
        one = Scalar.from_value(1.0, like=Tn)
        x = Tn / S0
        safe_x = where(tensile, x, one)

        ramp = where(tensile, pow(safe_x, beta), zero)
        Ndot = F_N * ramp * edot

        if v is None:
            return Ndot

        dramp_dx = where(tensile, beta * pow(safe_x, beta - 1.0), zero)
        dN_dTn = F_N * dramp_dx * edot / S0
        dN_dedot = F_N * ramp

        actions: dict[str, ChainRuleAction] = {
            "normal_traction": lambda V, c=dN_dTn: c * V,
            "strain_rate": lambda V, c=dN_dedot: c * V,
        }
        if "F_N" in self._promoted_params:
            dN_dFN = ramp * edot
            actions[self._promoted_params["F_N"].input_name] = lambda V, c=dN_dFN: c * V
        if "S0" in self._promoted_params:
            # d/dS0 of (Tn/S0)^beta = -beta/S0 * (Tn/S0)^beta
            dN_dS0 = -F_N * dramp_dx * edot * x / S0
            actions[self._promoted_params["S0"].input_name] = lambda V, c=dN_dS0: c * V
        return Ndot, self.apply_chain_rule(v, "cavity_density_rate", actions, output=Ndot)


__all__ = ["CavityNucleationRate"]
