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

"""Python-native mirror of the C++ ``NucleationFluxMagnitude`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar, exp


@register_native("NucleationFluxMagnitude")
class NucleationFluxMagnitude(Model):
    r"""Magnitude of the nucleation flux (excluding the Dirac delta term).

    Computes :math:`J = Z \, \beta \, N_0 \, \exp\!\bigl(-\Delta g / (k T)\bigr)`,
    where :math:`Z` is the Zeldovich factor, :math:`\beta` is the kinetic
    factor, :math:`\Delta g` is the nucleation barrier, :math:`T` is the
    temperature, :math:`N_0` is the nucleation site density, and :math:`k` is
    the Boltzmann constant.
    """

    hit = HitSchema(
        input("zeldovich_factor", Scalar, "Zeldovich factor"),
        input("kinetic_factor", Scalar, "Kinetic factor"),
        input("nucleation_barrier", Scalar, "Nucleation barrier"),
        input("temperature", Scalar, "Temperature"),
        output(
            "nucleation_flux_magnitude",
            Scalar,
            "Nucleation flux magnitude excluding the Dirac delta term",
        ),
        parameter(
            "nucleation_site_density",
            Scalar,
            "Nucleation site density",
            attr="N0",
            allow_nonlinear=True,
        ),
        parameter(
            "boltzmann_constant",
            Scalar,
            "Boltzmann constant",
            attr="k",
        ),
    )

    # ``from_hit`` auto-declares ``nucleation_site_density`` (stored as ``N0``)
    # and ``boltzmann_constant`` (stored as ``k``). Annotate so pyright sees
    # the typed wrappers that ``Model.__getattr__`` returns.
    N0: Scalar
    k: Scalar

    def forward(  # type: ignore[override]
        self,
        zeldovich_factor: Scalar,
        kinetic_factor: Scalar,
        nucleation_barrier: Scalar,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        Z = zeldovich_factor
        beta = kinetic_factor
        dg = nucleation_barrier
        T = temperature
        N0 = self._get_param("N0", nl_params, Scalar)
        k = self._get_param("k", nl_params, Scalar)

        # Forward: J = Z · β · N0 · exp(-Δg / (k T))
        kT = k * T
        exp_raw = exp(-dg / kT)
        prefactor = Z * beta * N0
        J = prefactor * exp_raw

        if v is None:
            return J

        # Differential pushforward. Linear coefficients (mirroring the
        # C++ ``set_value`` partials):
        #   ∂J/∂Z   = β · N0 · exp_raw            ⇒ action(V) = (β·N0·exp_raw) · V
        #   ∂J/∂β   = Z · N0 · exp_raw            ⇒ action(V) = (Z·N0·exp_raw) · V
        #   ∂J/∂Δg  = -prefactor · exp_raw / (kT) ⇒ action(V) = (-J/(kT)) · V
        #   ∂J/∂T   = prefactor · exp_raw · Δg / (k T²) ⇒ action(V) = (J·Δg/(k·T²)) · V
        #   ∂J/∂N0  = Z · β · exp_raw             ⇒ action(V) = (Z·β·exp_raw) · V
        # The N0 action only fires when N0 has been promoted to a nonlinear
        # input via the HIT ``[Models]`` cross-ref form.
        dJ_dZ = beta * N0 * exp_raw
        dJ_dbeta = Z * N0 * exp_raw
        dJ_ddg = -J / kT
        dJ_dT = J * dg / (k * T * T)

        actions: dict[str, ChainRuleAction] = {
            "zeldovich_factor": lambda V, c=dJ_dZ: c * V,
            "kinetic_factor": lambda V, c=dJ_dbeta: c * V,
            "nucleation_barrier": lambda V, c=dJ_ddg: c * V,
            "temperature": lambda V, c=dJ_dT: c * V,
        }

        N0_nlp = self._nl_params.get("N0")
        if N0_nlp is not None:
            dJ_dN0 = Z * beta * exp_raw
            actions[N0_nlp.input_name] = lambda V, c=dJ_dN0: c * V

        return J, self.apply_chain_rule(v, "nucleation_flux_magnitude", actions, output=J)


__all__ = ["NucleationFluxMagnitude"]
