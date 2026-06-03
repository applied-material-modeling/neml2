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

"""Python-native mirror of the C++ ``SFFKPrecipitationGrowthRate`` model."""

from __future__ import annotations

from typing import cast

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar


@register_native("SFFKPrecipitationGrowthRate")
class SFFKPrecipitationGrowthRate(Model):
    """Compute the SFFK precipitate growth rate."""

    hit = HitSchema(
        input("projected_diffusivity_sum", Scalar, "Projected diffusivity sum"),
        input("temperature", Scalar, "Temperature"),
        output("growth_rate", Scalar, "Precipitate growth rate per size bin"),
        parameter(
            "radius",
            Scalar,
            "Precipitate radius per size bin",
            attr="R",
            allow_nonlinear=True,
        ),
        parameter(
            "gibbs_free_energy_difference",
            Scalar,
            "Gibbs free energy difference",
            attr="dg",
            allow_nonlinear=True,
        ),
        parameter(
            "gas_constant",
            Scalar,
            "Gas constant",
            attr="R_g",
            allow_nonlinear=True,
        ),
    )
    # ``radius`` is a per-bin (sub_batch_ndim=1) Scalar while the other
    # parameters and the structural inputs may be scalar (sub_batch_ndim=0).
    # The bin-broadcast made by ``align_sub_batch`` makes site-i of
    # ``growth_rate`` depend on the single scalar value of any sub_batch_ndim=0
    # input/promoted parameter -- a cross-bin (dense) coupling rather than the
    # default per-site diagonal one. Declare every input pair as ``"dense"`` so
    # the composed-model resolver picks the conservative (BLOCK -> DENSE
    # fallback) sparsity flag. Per-bin inputs still produce a strictly
    # bin-diagonal action; the flag is purely an upper bound used by the
    # system assembler.
    list_deriv = {
        ("growth_rate", "projected_diffusivity_sum"): "dense",
        ("growth_rate", "temperature"): "dense",
        ("growth_rate", "radius"): "dense",
        ("growth_rate", "gibbs_free_energy_difference"): "dense",
        ("growth_rate", "gas_constant"): "dense",
    }

    # ``from_hit`` auto-declares the three parameters (stored as R, dg, R_g).
    # Annotate so pyright sees the typed wrappers that ``Model.__getattr__``
    # returns; runtime resolution is via ``_get_param`` so the same code path
    # handles both static and nl-promoted values.
    R: Scalar
    dg: Scalar
    R_g: Scalar

    def forward(  # type: ignore[override]
        self,
        projected_diffusivity_sum: Scalar,
        temperature: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        proj_sum = projected_diffusivity_sum
        T = temperature
        R = self._get_param("R", nl_params, Scalar)
        dg = self._get_param("dg", nl_params, Scalar)
        R_g = self._get_param("R_g", nl_params, Scalar)

        # Forward: R_dot = dg / (R * R_g * T * proj_sum). Typed Scalar algebra
        # end-to-end; align_sub_batch transparently lifts sub_batch_ndim=0
        # parameters/inputs up to the per-bin sub_batch_ndim of R, matching
        # the C++ ``set_value``.
        denom = R * R_g * T * proj_sum
        rate = dg / denom

        if v is None:
            return rate

        # Differential pushforward. The forward is a simple rational
        # function of (proj_sum, T, R, dg, R_g); every action is a closed-form
        # coefficient times the input tangent. Wrapper algebra handles the
        # scalar-vs-per-bin broadcast on the tangent the same way it does on
        # the forward, so each action stays pure typed Scalar arithmetic -- no
        # diagonal embedding, no Jacobian materialization.
        #
        #   d R_dot / d proj_sum = -rate / proj_sum
        #   d R_dot / d T        = -rate / T
        #   d R_dot / d R        = -rate / R
        #   d R_dot / d dg       =  1 / denom
        #   d R_dot / d R_g      = -rate / R_g
        actions: dict[str, ChainRuleAction] = {}

        d_rate_dsum = -rate / proj_sum

        def sum_action(V: Scalar) -> Scalar:
            return cast(Scalar, d_rate_dsum * V)

        actions["projected_diffusivity_sum"] = sum_action

        d_rate_dT = -rate / T

        def T_action(V: Scalar) -> Scalar:
            return cast(Scalar, d_rate_dT * V)

        actions["temperature"] = T_action

        R_nlp = self._nl_params.get("R")
        if R_nlp is not None:
            d_rate_dR = -rate / R

            def R_action(V: Scalar) -> Scalar:
                return cast(Scalar, d_rate_dR * V)

            actions[R_nlp.input_name] = R_action

        dg_nlp = self._nl_params.get("dg")
        if dg_nlp is not None:
            inv_denom = 1.0 / denom

            def dg_action(V: Scalar) -> Scalar:
                return cast(Scalar, inv_denom * V)

            actions[dg_nlp.input_name] = dg_action

        R_g_nlp = self._nl_params.get("R_g")
        if R_g_nlp is not None:
            d_rate_dRg = -rate / R_g

            def R_g_action(V: Scalar) -> Scalar:
                return cast(Scalar, d_rate_dRg * V)

            actions[R_g_nlp.input_name] = R_g_action

        return rate, self.apply_chain_rule(v, "growth_rate", actions, output=rate)


__all__ = ["SFFKPrecipitationGrowthRate"]
