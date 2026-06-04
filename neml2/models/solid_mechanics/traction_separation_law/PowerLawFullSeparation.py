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

"""Python-native mirror of the C++ ``PowerLawFullSeparation`` model."""

from __future__ import annotations

import torch

from ....chain_rule import ChainRuleAction, ChainRuleDict
from ....factory import register_neml2_object
from ....model import Model
from ....schema import HitSchema, input, output, parameter
from ....types import Scalar, gt, pow, where


@register_neml2_object("PowerLawFullSeparation")
class PowerLawFullSeparation(Model):
    r"""Mixed-mode full (failure) separation under the Alfano-Crisfield power-law criterion.
    Opening:
    $\delta_f = (2(1+\beta^2)/(K \delta_c)) [(1/G_{Ic})^\eta + (\beta^2/G_{IIc})^\eta]^{-1/\eta}$,
    where $\delta_c$ is the critical
    separation supplied by an upstream `CamanhoDavilaCriticalSeparation`. Compression:
    $\delta_f = 2 G_{IIc}/S$.
    """

    hit = HitSchema(
        input(
            "normal_separation",
            Scalar,
            "Normal separation (typically the Macaulay-positive part of the normal jump; "
            "used only to determine the opening / compression branch)",
        ),
        output("full_separation", Scalar, "Full (failure) separation"),
        parameter(
            "mode_mixity",
            Scalar,
            "Mode-mixity ratio. May be wired to an upstream `ModeMixity` (nonlinear-capable).",
            attr="beta",
            allow_nonlinear=True,
        ),
        parameter(
            "critical_separation",
            Scalar,
            "Critical (damage-onset) separation. May be wired to an upstream "
            "`CamanhoDavilaCriticalSeparation` (nonlinear-capable).",
            attr="delta_c",
            allow_nonlinear=True,
        ),
        parameter("penalty_stiffness", Scalar, "Penalty stiffness", attr="K"),
        parameter(
            "mode_I_fracture_toughness",
            Scalar,
            "Mode I critical energy release rate",
            attr="GIc",
        ),
        parameter(
            "mode_II_fracture_toughness",
            Scalar,
            "Mode II critical energy release rate",
            attr="GIIc",
        ),
        parameter(
            "shear_strength",
            Scalar,
            "Shear strength (used in compression branch)",
            attr="S",
        ),
        parameter("eta", Scalar, "Power-law exponent", attr="eta"),
    )

    # ``from_hit`` auto-declares each parameter via ``declare_typed_parameter``.
    # Annotate so pyright sees the typed wrappers ``Model.__getattr__`` returns.
    beta: Scalar
    delta_c: Scalar
    K: Scalar
    GIc: Scalar
    GIIc: Scalar
    S: Scalar
    eta: Scalar

    def forward(  # type: ignore[override]
        self,
        dn: Scalar,
        *nl_params: Scalar,
        v: ChainRuleDict | None = None,
    ) -> Scalar | tuple[Scalar, ChainRuleDict]:
        # ``beta`` and ``delta_c`` are nonlinear-capable; route through
        # ``_get_param`` so the same code path works whether they remained
        # static parameters or were promoted to runtime nl inputs.
        beta = self._get_param("beta", nl_params, Scalar)
        delta_c = self._get_param("delta_c", nl_params, Scalar)
        K = self._get_param("K", nl_params, Scalar)
        GIc = self._get_param("GIc", nl_params, Scalar)
        GIIc = self._get_param("GIIc", nl_params, Scalar)
        S = self._get_param("S", nl_params, Scalar)
        eta = self._get_param("eta", nl_params, Scalar)

        # Match the C++ ``machine_precision(dtype)`` regularizer used to keep
        # ``pow(pow_base, eta)`` differentiable at ``beta == 0``.
        eps = torch.finfo(dn.data.dtype).eps

        # Opening branch (delta_n > 0).
        beta_sq = beta * beta
        pow_base = beta_sq / GIIc + eps
        Gc_mixed = pow(1.0 / GIc, eta) + pow(pow_base, eta)
        Gc_term = pow(Gc_mixed, -1.0 / eta)
        prefactor = (2.0 + 2.0 * beta_sq) / (K * delta_c)
        delta_final_open = prefactor * Gc_term
        # Compression branch (delta_n <= 0): pure-shear closed form.
        delta_final_default = 2.0 * GIIc / S

        # Branch select. The mask is treated as constant w.r.t. ``dn``
        # (matches C++ ``.detach()``): ``normal_separation`` carries no
        # action and pushes forward to structural zero.
        pos_mask = gt(dn, 0.0)
        delta_f = where(pos_mask, delta_final_open, delta_final_default)

        if v is None:
            return delta_f

        # -------- D-062 pushforward.
        #
        # The forward depends on ``normal_separation`` only through the
        # detached branch mask, so its action is structural zero (omitted
        # from ``actions`` -> ``apply_chain_rule`` treats it as zero). The
        # two nonlinear-capable parameters ``beta`` and ``delta_c`` carry
        # closed-form partials, emitted only when promoted to runtime
        # inputs (mode 3/4).
        actions: dict[str, ChainRuleAction] = {}

        delta_c_nlp = self._nl_params.get("delta_c")
        if delta_c_nlp is not None:
            # Opening: d(delta_f)/d(delta_c) = -delta_final_open / delta_c.
            # Compression branch is independent of delta_c.
            zero = Scalar.from_value(0.0, like=delta_f)
            ddf_dinit_open = -delta_final_open / delta_c
            ddf_dinit = where(pos_mask, ddf_dinit_open, zero)
            actions[delta_c_nlp.input_name] = lambda V, c=ddf_dinit: c * V

        beta_nlp = self._nl_params.get("beta")
        if beta_nlp is not None:
            zero = Scalar.from_value(0.0, like=delta_f)
            # dGc_mixed/dbeta = eta (beta^2/GIIc + eps)^(eta-1) (2 beta / GIIc)
            dGc_mixed_dbeta = eta * pow(pow_base, eta - 1.0) * (2.0 * beta / GIIc)
            # dGc_term/dbeta = (-1/eta) Gc_mixed^(-1/eta - 1) dGc_mixed/dbeta
            dGc_term_dbeta = (-1.0 / eta) * pow(Gc_mixed, -1.0 / eta - 1.0) * dGc_mixed_dbeta
            dprefactor_dbeta = 4.0 * beta / (K * delta_c)
            ddf_dbeta_open = dprefactor_dbeta * Gc_term + prefactor * dGc_term_dbeta
            ddf_dbeta = where(pos_mask, ddf_dbeta_open, zero)
            actions[beta_nlp.input_name] = lambda V, c=ddf_dbeta: c * V

        return delta_f, self.apply_chain_rule(v, "full_separation", actions, output=delta_f)


__all__ = ["PowerLawFullSeparation"]
