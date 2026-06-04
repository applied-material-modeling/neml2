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

"""Python-native mirror of the C++ ``PerSlipForestDislocationEvolution`` leaf."""

from __future__ import annotations

from ....chain_rule import ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, derived_output, input, parameter
from ....types import Scalar, abs, sign, sqrt


@register_neml2_object("PerSlipForestDislocationEvolution")
class PerSlipForestDislocationEvolution(Model):
    r"""Standard forest hardening model per slip system defined by
    $\dot{\rho}_i = (k_1 \sqrt{\rho_i} - k_2 \rho_i) \left| \dot{\gamma}_i \right|$.
    """

    # Variable names match the C++ defaults.
    # The output ``dislocation_density_rate`` is derived from the
    # ``dislocation_density`` input, mirroring the C++
    # ``rate_name(_rho.name())`` convention.
    hit = HitSchema(
        input("dislocation_density", Scalar, "Per-slip dislocation density"),
        input("slip_rates", Scalar, "Per-slip system slip rates"),
        derived_output("dislocation_density", Scalar, attr="_rho_rate", suffix="_rate"),
        parameter("k1", Scalar, "Hardening coefficient", attr="k1", allow_nonlinear=True),
        parameter("k2", Scalar, "Recovery coefficient", attr="k2", allow_nonlinear=True),
    )

    # ``from_hit`` auto-declares the two parameters and the derived
    # ``dislocation_density_rate`` output name (stored on ``self._rho_rate``).
    k1: Scalar
    k2: Scalar
    _rho_rate: str

    def forward(  # type: ignore[override]
        self,
        rho: Scalar,
        gamma_dot: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # Mirrors ``PerSlipForestDislocationEvolution::set_value`` in the C++ source.
        k1 = self._get_param("k1", nl_params, Scalar)
        k2 = self._get_param("k2", nl_params, Scalar)

        sqrt_rho = sqrt(rho)
        abs_gamma = abs(gamma_dot)
        hardening = k1 * sqrt_rho - k2 * rho
        rho_dot = hardening * abs_gamma

        if v is None:
            return rho_dot

        # Differential pushforward. Closed-form coefficients of typed
        # Scalars; no Jacobian materialised. Derivation matches the dense
        # C++ jacobian rows in ``set_value``:
        #   d rho_dot / d rho       = (k1 * 0.5 / sqrt(rho) - k2) * |gamma_dot|
        #   d rho_dot / d gamma_dot = hardening * sign(gamma_dot)
        #   d rho_dot / d k1        = sqrt(rho) * |gamma_dot|
        #   d rho_dot / d k2        = -rho * |gamma_dot|
        d_drho = (0.5 * k1 / sqrt_rho - k2) * abs_gamma
        d_dgamma = hardening * sign(gamma_dot)

        actions = {
            "dislocation_density": lambda V, c=d_drho: c * V,
            "slip_rates": lambda V, c=d_dgamma: c * V,
        }
        if "k1" in self._nl_params:
            actions[self._nl_params["k1"].input_name] = lambda V, c=sqrt_rho * abs_gamma: c * V
        if "k2" in self._nl_params:
            actions[self._nl_params["k2"].input_name] = lambda V, c=-rho * abs_gamma: c * V

        return rho_dot, self.apply_chain_rule(v, self._rho_rate, actions, output=rho_dot)
