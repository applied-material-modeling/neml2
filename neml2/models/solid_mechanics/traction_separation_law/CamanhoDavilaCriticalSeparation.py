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

"""Python-native mirror of the C++ ``CamanhoDavilaCriticalSeparation`` model."""

from __future__ import annotations

import torch

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, sqrt, where


@register_neml2_object("CamanhoDavilaCriticalSeparation")
class CamanhoDavilaCriticalSeparation(Model):
    r"""Camanho-Davila mixed-mode critical (damage-onset) separation. Opening branch:
    $\delta_c = \delta_{n0} \delta_{s0} \sqrt{1+\beta^2} / \sqrt{\delta_{s0}^2 + \beta^2 \delta_{n0}^2}$
    with $\delta_{n0} = N/K$ and $\delta_{s0} = S/K$.
    Compression branch: $\delta_c = \delta_{s0}$.
    """  # noqa: E501

    hit = HitSchema(
        input(
            "normal_separation",
            Scalar,
            "Normal separation (typically the Macaulay-positive part of the normal jump; "
            "used only to determine the opening / compression branch)",
        ),
        output("critical_separation", Scalar, "Critical (damage-onset) separation"),
        parameter(
            "mode_mixity",
            Scalar,
            "Mode-mixity ratio. May be wired to an upstream `ModeMixity` (nonlinear-capable).",
            attr="beta",
            allow_nonlinear=True,
        ),
        parameter("penalty_stiffness", Scalar, "Penalty stiffness K", attr="K"),
        parameter("normal_strength", Scalar, "Tensile (normal) strength N", attr="N"),
        parameter("shear_strength", Scalar, "Shear strength S", attr="S"),
    )

    # ``from_hit`` auto-declares each parameter via ``declare_typed_parameter``.
    # Annotate so pyright sees the typed wrappers ``Model.__getattr__`` returns.
    beta: Scalar
    K: Scalar
    N: Scalar
    S: Scalar

    def forward(  # type: ignore[override]
        self,
        dn: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # ``beta`` is nonlinear-capable; route through ``_get_param`` so the
        # same code path works whether it remained a static parameter or got
        # promoted to a runtime nl input. ``K``, ``N``, ``S`` are
        # allow_nonlinear=False on the C++ side (no nl-Jacobian), but the
        # ``_get_param`` resolver works uniformly for static parameters too.
        beta = self._get_param("beta", nl_params, Scalar)
        K = self._get_param("K", nl_params, Scalar)
        N = self._get_param("N", nl_params, Scalar)
        S = self._get_param("S", nl_params, Scalar)

        # Match the C++ ``machine_precision(_dn.scalar_type())`` regularizer
        # used to keep the inner ``sqrt`` differentiable at ``beta == 0``.
        eps = torch.finfo(dn.data.dtype).eps

        delta_n0 = N / K
        delta_s0 = S / K

        beta_sq = beta * beta
        delta_mixed_sq = delta_s0 * delta_s0 + beta_sq * delta_n0 * delta_n0 + eps
        delta_mixed = sqrt(delta_mixed_sq)
        delta_init_open = delta_n0 * delta_s0 * sqrt(1.0 + beta_sq) / delta_mixed

        # Branch select. The mask is treated as constant w.r.t. ``dn``
        # (matches C++ ``.detach()``): ``normal_separation`` carries no
        # action and pushes forward to structural zero.
        pos_mask = gt(dn, 0.0)
        delta_c = where(pos_mask, delta_init_open, delta_s0)

        if v is None:
            return delta_c

        # -------- D-062 pushforward.
        #
        # The forward depends on ``normal_separation`` only through the
        # detached branch mask, so its action is structural zero (omitted
        # from ``actions`` -> ``apply_chain_rule`` treats it as zero). The
        # ``beta`` partial is emitted only when promoted to a runtime input
        # (mode 3/4). ``K``, ``N``, ``S`` are allow_nonlinear=False so no
        # action is emitted for them.
        actions: dict[str, ChainRuleAction] = {}

        beta_nlp = self._nl_params.get("beta")
        if beta_nlp is not None:
            zero = Scalar.from_value(0.0, like=delta_c)
            # Opening: d(delta_c)/d(beta) =
            #   delta_init_open * beta * [1/(1+beta_sq) - delta_n0^2/delta_mixed_sq].
            # Compression branch is independent of beta.
            ddc_dbeta_open = (
                delta_init_open
                * beta
                * (1.0 / (1.0 + beta_sq) - delta_n0 * delta_n0 / delta_mixed_sq)
            )
            ddc_dbeta = where(pos_mask, ddc_dbeta_open, zero)
            actions[beta_nlp.input_name] = lambda V, c=ddc_dbeta: c * V

        return delta_c, self.apply_chain_rule(v, "critical_separation", actions, output=delta_c)


__all__ = ["CamanhoDavilaCriticalSeparation"]
