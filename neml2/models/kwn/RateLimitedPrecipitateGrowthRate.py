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

"""Python-native mirror of the C++ ``RateLimitedPrecipitateGrowthRate`` model."""

from __future__ import annotations

from ...chain_rule import ChainRuleAction, ChainRuleDict
from ...factory import register_native
from ...model import Model
from ...schema import HitSchema, input, output, parameter
from ...types import Scalar


@register_native("RateLimitedPrecipitateGrowthRate")
class RateLimitedPrecipitateGrowthRate(Model):
    """Compute the rate-limited precipitate growth rate for a single species."""

    hit = HitSchema(
        input("current_concentration", Scalar, "Current concentration in solution"),
        output("growth_rate", Scalar, "Precipitate growth rate per size bin"),
        parameter(
            "radius",
            Scalar,
            "Precipitate radius per size bin",
            attr="R",
            allow_nonlinear=True,
        ),
        parameter(
            "equilibrium_concentration",
            Scalar,
            "Equilibrium concentration in solution",
            attr="x_eq",
            allow_nonlinear=True,
        ),
        parameter(
            "concentration_difference",
            Scalar,
            "Concentration difference between precipitate and equilibrium",
            attr="dx",
            allow_nonlinear=True,
        ),
        parameter(
            "diffusivity",
            Scalar,
            "Species diffusivity in solution",
            attr="D",
            allow_nonlinear=True,
        ),
    )
    # ``radius`` is a per-bin (sub_batch_ndim=1) Scalar while the other
    # parameters and the structural input may be scalar (sub_batch_ndim=0). The
    # bin-broadcast made by ``align_sub_batch`` makes site-i of ``growth_rate``
    # depend on the single scalar value of any sub_batch_ndim=0 input/promoted
    # parameter -- a cross-bin (dense) coupling rather than the default
    # per-site diagonal one. Declare every input pair as ``"dense"`` so the
    # composed-model resolver picks the conservative (BLOCK -> DENSE fallback)
    # sparsity flag. Per-bin inputs still produce a strictly bin-diagonal
    # action; the flag is purely an upper bound used by the system assembler.
    list_deriv = {
        ("growth_rate", "current_concentration"): "dense",
        ("growth_rate", "radius"): "dense",
        ("growth_rate", "equilibrium_concentration"): "dense",
        ("growth_rate", "concentration_difference"): "dense",
        ("growth_rate", "diffusivity"): "dense",
    }

    # ``from_hit`` auto-declares the four parameters (stored as R, x_eq, dx, D).
    # Annotate so pyright sees the typed wrappers that ``Model.__getattr__``
    # returns; runtime resolution is via ``_get_param`` so the same code path
    # handles both static and nl-promoted values.
    R: Scalar
    x_eq: Scalar
    dx: Scalar
    D: Scalar

    def forward(  # type: ignore[override]
        self,
        x: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        R = self._get_param("R", nl_params, Scalar)
        x_eq = self._get_param("x_eq", nl_params, Scalar)
        dx = self._get_param("dx", nl_params, Scalar)
        D = self._get_param("D", nl_params, Scalar)

        # Forward: R_dot = (D / R) * (x - x_eq) / dx. Typed Scalar algebra
        # end-to-end; align_sub_batch transparently lifts sub_batch_ndim=0
        # parameters (or input) up to the per-bin sub_batch_ndim of R, matching
        # the C++ ``intmd_expand(nbin)`` calls in ``set_value``.
        numer = x - x_eq
        coef = D / R
        rate = coef * numer / dx

        if v is None:
            return rate

        # Differential pushforward. The forward is a simple rational
        # function of (x, R, x_eq, dx, D); every action is a closed-form
        # coefficient times the input tangent. Wrapper algebra handles the
        # scalar-vs-per-bin broadcast on the tangent the same way it did on
        # the forward, so each action stays pure typed Scalar arithmetic -- no
        # diagonal embedding, no Jacobian materialization.
        #
        #   d R_dot / d x    = D / (R * dx)        =  coef / dx
        #   d R_dot / d x_eq = -D / (R * dx)       = -coef / dx
        #   d R_dot / d dx   = -D * numer / (R * dx^2)
        #   d R_dot / d R    = -D * numer / (R^2 * dx)
        #   d R_dot / d D    = numer / (R * dx)
        actions: dict[str, ChainRuleAction] = {}

        d_rate_dx = coef / dx

        def x_action(V: Scalar) -> Scalar:
            return d_rate_dx * V

        actions["current_concentration"] = x_action

        x_eq_nlp = self._nl_params.get("x_eq")
        if x_eq_nlp is not None:
            d_rate_dxeq = -coef / dx

            def x_eq_action(V: Scalar) -> Scalar:
                return d_rate_dxeq * V

            actions[x_eq_nlp.input_name] = x_eq_action

        dx_nlp = self._nl_params.get("dx")
        if dx_nlp is not None:
            d_rate_ddx = -coef * numer / (dx * dx)

            def dx_action(V: Scalar) -> Scalar:
                return d_rate_ddx * V

            actions[dx_nlp.input_name] = dx_action

        R_nlp = self._nl_params.get("R")
        if R_nlp is not None:
            d_rate_dR = -coef * numer / (dx * R)

            def R_action(V: Scalar) -> Scalar:
                return d_rate_dR * V

            actions[R_nlp.input_name] = R_action

        D_nlp = self._nl_params.get("D")
        if D_nlp is not None:
            d_rate_dD = numer / (R * dx)

            def D_action(V: Scalar) -> Scalar:
                return d_rate_dD * V

            actions[D_nlp.input_name] = D_action

        return rate, self.apply_chain_rule(v, "growth_rate", actions, output=rate)


__all__ = ["RateLimitedPrecipitateGrowthRate"]
