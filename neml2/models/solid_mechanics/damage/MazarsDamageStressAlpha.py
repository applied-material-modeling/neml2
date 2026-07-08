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

"""Python-native port of the v2 C++ ``MazarsDamageStressAlpha`` model."""

from __future__ import annotations

from ....factory import register_neml2_object
from ....schema import HitSchema, input, output, parameter
from ....types import R2, SR2, Scalar, Vec, exp, heaviside, inner, macaulay
from ....types.functions import r2_from_sr2
from ....types.linalg import diag, eigh, transpose
from ...chain_rule import ChainRuleDict
from ...model import Model

# Numerical guard for 1/ε̃² when no positive principal strains exist
# (e.g., uniaxial-strain pure compression, hydrostatic compression). Matches
# the v2 C++ implementation's regularizer to the bit.
_EPS_FLOOR = 1.0e-30


@register_neml2_object("MazarsDamageStressAlpha")
class MazarsDamageStressAlpha(Model):
    r"""Mazars (1986) scalar damage law with the STRESS-BASED :math:`\alpha`
    weighting from Section 3.5.1 of the original paper.

    Distinguished from a strain-magnitude :math:`\alpha` formulation
    (``MazarsDamage``) by computing the tension/compression mixing weights
    from the effective stress's principal-sign decomposition, mapped back
    to a strain contribution via the isotropic compliance:

    .. math::
        \boldsymbol{\sigma}_t = \langle \boldsymbol{\sigma}_i \rangle_+, \qquad
        \boldsymbol{\sigma}_c = -\langle -\boldsymbol{\sigma}_i \rangle_+
    .. math::
        \boldsymbol{\varepsilon}_t = \frac{(1+\nu)}{E} \boldsymbol{\sigma}_t
        - \frac{\nu}{E} \mathrm{tr}(\boldsymbol{\sigma}_t)
    .. math::
        \alpha_t = \frac{\sum_i \varepsilon_{t,i} \langle \varepsilon_i \rangle_+}
                        {\sum_i \langle \varepsilon_i \rangle_+^2}, \qquad
        \alpha_c = \frac{\sum_i \varepsilon_{c,i} \langle \varepsilon_i \rangle_+}
                        {\sum_i \langle \varepsilon_i \rangle_+^2}

    The damage laws are the standard Mazars exponentials:

    .. math::
        D_{t,c}(\tilde\varepsilon_{\max}) = 1 - \frac{\varepsilon_{d0}
        (1 - A_{t,c})}{\tilde\varepsilon_{\max}}
        - A_{t,c}\, e^{-B_{t,c} (\tilde\varepsilon_{\max} - \varepsilon_{d0})}

    blended as :math:`D = \alpha_t D_t + \alpha_c D_c`.

    The compliance constants ``E`` and :math:`\nu` parameterize the back-map
    and MUST match the upstream :class:`LinearIsotropicElasticity` block. The
    input file is responsible for that consistency -- no run-time check.

    Reference: Mazars (1986), Engineering Fracture Mechanics 25(5-6):729-737,
    Section 3.5.1 (:math:`\alpha` weighting) and Section 3.5.2 (damage laws).
    """

    hit = HitSchema(
        input("equivalent_strain", Scalar, "Mazars equivalent-strain history max"),
        input(
            "strain",
            SR2,
            "Total strain (SR2). Eigendecomposed for the principal-strain "
            "alignment weights in the alpha formula.",
        ),
        input(
            "effective_stress",
            SR2,
            "Effective (undamaged) stress sigma_tilde = C : strain (SR2). "
            "Eigendecomposed and split into +/- parts for alpha.",
        ),
        output("damage", Scalar, "Output damage variable D"),
        parameter(
            "eps_d0", Scalar, "Damage onset equivalent strain", attr="eps_d0", allow_nonlinear=True
        ),
        parameter(
            "A_t", Scalar, "Tension damage law shape parameter", attr="A_t", allow_nonlinear=True
        ),
        parameter(
            "B_t", Scalar, "Tension damage law rate parameter", attr="B_t", allow_nonlinear=True
        ),
        parameter(
            "A_c",
            Scalar,
            "Compression damage law shape parameter",
            attr="A_c",
            allow_nonlinear=True,
        ),
        parameter(
            "B_c", Scalar, "Compression damage law rate parameter", attr="B_c", allow_nonlinear=True
        ),
        parameter(
            "E", Scalar, "Young's modulus (for compliance back-map)", attr="E", allow_nonlinear=True
        ),
        parameter(
            "nu", Scalar, "Poisson ratio (for compliance back-map)", attr="nu", allow_nonlinear=True
        ),
    )

    eps_d0: Scalar
    A_t: Scalar
    B_t: Scalar
    A_c: Scalar
    B_c: Scalar
    E: Scalar
    nu: Scalar

    def forward(  # type: ignore[override]
        self,
        equivalent_strain: Scalar,
        strain: SR2,
        effective_stress: SR2,
        *nl_params,
        v: ChainRuleDict | None = None,
    ):
        # ============================================================
        # 0. Read parameters (static slots or nonlinear-promoted inputs)
        # ============================================================
        e0 = self._get_param("eps_d0", nl_params, Scalar)
        At = self._get_param("A_t", nl_params, Scalar)
        Bt = self._get_param("B_t", nl_params, Scalar)
        Ac = self._get_param("A_c", nl_params, Scalar)
        Bc = self._get_param("B_c", nl_params, Scalar)
        E_ = self._get_param("E", nl_params, Scalar)
        nu = self._get_param("nu", nl_params, Scalar)

        # ============================================================
        # 1. Damage laws D_t and D_c — Mazars 1986 §3.5.2
        # ============================================================
        # Clamp ε_max ↑ to ε_d0 via the Macaulay identity:
        #   max(ε_max, ε_d0) = ⟨ε_max − ε_d0⟩ + ε_d0
        # At ε_max = ε_d0 the formula gives D = 0 identically; below the
        # threshold we evaluate at ε_d0 itself, also giving D = 0.
        eps_above = macaulay(equivalent_strain - e0)
        eps_eval = eps_above + e0

        Dt_raw = 1.0 - e0 * (1.0 - At) / eps_eval - At * exp(-Bt * eps_above)
        Dc_raw = 1.0 - e0 * (1.0 - Ac) / eps_eval - Ac * exp(-Bc * eps_above)
        # Lower-clamp to 0; the formula is bounded above by 1 for any
        # physical parameter set so no upper cap is needed.
        Dt = macaulay(Dt_raw)
        Dc = macaulay(Dc_raw)

        # ============================================================
        # 2. Stress-based α weights — Mazars 1986 §3.5.1
        # ============================================================
        # (a) Principal stresses and directions from σ̃.
        sig_principals, V_sig = eigh(effective_stress)

        # (b) Split into ± principal parts.
        #     σ_t,i = ⟨σ_i⟩      (positive part)
        #     σ_c,i = -⟨-σ_i⟩    (negative part, with sign preserved)
        sig_t = macaulay(sig_principals)
        sig_c = -macaulay(-sig_principals)

        # (c) Traces. inner(v, ones) is the typed way to base-sum a Vec.
        ones3 = Vec.ones()
        tr_sig_t = inner(sig_t, ones3)
        tr_sig_c = inner(sig_c, ones3)

        # (d) Isotropic compliance back-map (principal form of
        #     ε = (1+ν)/E · σ − ν/E · tr(σ) I):
        fac_diag = (1.0 + nu) / E_
        fac_tr = nu / E_
        eps_t = fac_diag * sig_t - fac_tr * tr_sig_t
        eps_c = fac_diag * sig_c - fac_tr * tr_sig_c

        # (e) Principal strains and Macaulay-positive part.
        eps_principals, V_eps = eigh(strain)
        eps_pos = macaulay(eps_principals)

        # ε̃² = Σ ⟨ε_i⟩²
        eps_tilde_sq = inner(eps_pos, eps_pos)

        # Safe divide so 0/0 → 0 when no positive principal strains. Use
        # v2's exact regularizer for bit-identical primal output:
        # den_safe = ⟨ε̃² − ε⟩ + ε ≡ max(ε̃², ε).
        den_safe = macaulay(eps_tilde_sq - _EPS_FLOOR) + _EPS_FLOOR

        a_t = inner(eps_t, eps_pos) / den_safe
        a_c = inner(eps_c, eps_pos) / den_safe

        # ============================================================
        # 3. Final damage
        # ============================================================
        damage = a_t * Dt + a_c * Dc

        if v is None:
            return damage

        # ============================================================
        # 4. Closed-form JVPs (Phase F derivation; see RESEARCH_LOG §F)
        # ============================================================
        # Pre-compute reusable factors once, capture them by default-arg in
        # each closure (avoiding the late-binding pitfall).
        H_em = heaviside(equivalent_strain - e0)
        H_Dt = heaviside(Dt_raw)
        H_Dc = heaviside(Dc_raw)
        exp_Bt = exp(-Bt * eps_above)
        exp_Bc = exp(-Bc * eps_above)
        eps_eval_sq = eps_eval * eps_eval
        H_eps = heaviside(eps_principals)
        H_sig_pos = heaviside(sig_principals)
        H_sig_neg = heaviside(-sig_principals)
        tr_eps_pos = inner(eps_pos, ones3)

        # ---- Input JVPs ----

        # ∂D/∂ε_max — only Dt, Dc depend on ε_max.
        # ∂D_t/∂ε_max = H(Dt_raw) · H(ε_max−ε_d0)
        #               · [ε_d0 (1−A_t)/ε_eval² + A_t B_t exp(−B_t eps_above)]
        # (and analogous for D_c)
        dDt_dem = H_Dt * H_em * (e0 * (1.0 - At) / eps_eval_sq + At * Bt * exp_Bt)
        dDc_dem = H_Dc * H_em * (e0 * (1.0 - Ac) / eps_eval_sq + Ac * Bc * exp_Bc)
        dD_dem = a_t * dDt_dem + a_c * dDc_dem

        # ∂D/∂strain — only the α weights depend on strain.
        # ∂α_t/∂ε : V_S = (1/den_safe)
        #                · inner(ε_t · H_eps − 2 α_t ε_pos, diag(Vᵀ V_S V))
        # Combined: ∂D/∂ε : V_S = (1/den_safe)
        #   · inner((D_t ε_t + D_c ε_c) H_eps − 2 D ε_pos, diag(Vᵀ V_S V))
        strain_coeff = (Dt * eps_t + Dc * eps_c) * H_eps - 2.0 * damage * eps_pos

        def strain_action(
            V_S: SR2,
            V_eps: R2 = V_eps,
            strain_coeff: Vec = strain_coeff,
            den_safe: Scalar = den_safe,
        ) -> Scalar:
            V_S_full = r2_from_sr2(V_S)
            diag_term = diag(transpose(V_eps) @ V_S_full @ V_eps)
            return inner(strain_coeff, diag_term) / den_safe

        # ∂D/∂σ̃ — only the α weights depend on stress.
        # Using self-adjointness of the isotropic-compliance operator
        # C(v) = fac_diag·v − fac_tr·sum(v)·1:
        #   inner(ε_pos, ∂ε_t/∂σ : V_σ) = inner(C(ε_pos), ∂σ_t/∂σ : V_σ)
        # ∂σ_t,i/∂σ : V_σ = H_sig_pos_i · (s_iᵀ V_σ s_i)
        # ∂σ_c,i/∂σ : V_σ = H_sig_neg_i · (s_iᵀ V_σ s_i)
        # Combined: ∂D/∂σ : V_σ = (1/den_safe)
        #   · inner(C(ε_pos) · (D_t H_sig_pos + D_c H_sig_neg),
        #           diag(V_sigᵀ V_σ V_sig))
        eps_pos_compliance = fac_diag * eps_pos - fac_tr * tr_eps_pos
        stress_coeff = eps_pos_compliance * (Dt * H_sig_pos + Dc * H_sig_neg)

        def stress_action(
            V_sigma: SR2,
            V_sig: R2 = V_sig,
            stress_coeff: Vec = stress_coeff,
            den_safe: Scalar = den_safe,
        ) -> Scalar:
            V_sigma_full = r2_from_sr2(V_sigma)
            diag_sig = diag(transpose(V_sig) @ V_sigma_full @ V_sig)
            return inner(stress_coeff, diag_sig) / den_safe

        actions: dict = {
            "equivalent_strain": (lambda V, c=dD_dem: c * V),
            "strain": strain_action,
            "effective_stress": stress_action,
        }

        # ---- Parameter JVPs (only when promoted to nonlinear inputs) ----

        # ∂D/∂ε_d0:
        #   ∂D_t_raw/∂ε_d0 = −H_em · [(1−A_t)/ε_eval + A_t B_t exp(−B_t eps_above)]
        if "eps_d0" in self._nl_params:
            dDt_de0 = H_Dt * (-H_em) * ((1.0 - At) / eps_eval + At * Bt * exp_Bt)
            dDc_de0 = H_Dc * (-H_em) * ((1.0 - Ac) / eps_eval + Ac * Bc * exp_Bc)
            dD_de0 = a_t * dDt_de0 + a_c * dDc_de0
            actions[self._nl_params["eps_d0"].input_name] = lambda V, c=dD_de0: c * V

        # ∂D/∂A_t = α_t · H(Dt_raw) · [ε_d0/ε_eval − exp(−B_t eps_above)]
        if "A_t" in self._nl_params:
            dDt_dAt = H_Dt * (e0 / eps_eval - exp_Bt)
            dD_dAt = a_t * dDt_dAt
            actions[self._nl_params["A_t"].input_name] = lambda V, c=dD_dAt: c * V

        # ∂D/∂B_t = α_t · H(Dt_raw) · A_t · eps_above · exp(−B_t eps_above)
        if "B_t" in self._nl_params:
            dDt_dBt = H_Dt * At * eps_above * exp_Bt
            dD_dBt = a_t * dDt_dBt
            actions[self._nl_params["B_t"].input_name] = lambda V, c=dD_dBt: c * V

        # ∂D/∂A_c and ∂D/∂B_c — symmetric to A_t / B_t but with D_c / α_c.
        if "A_c" in self._nl_params:
            dDc_dAc = H_Dc * (e0 / eps_eval - exp_Bc)
            dD_dAc = a_c * dDc_dAc
            actions[self._nl_params["A_c"].input_name] = lambda V, c=dD_dAc: c * V

        if "B_c" in self._nl_params:
            dDc_dBc = H_Dc * Ac * eps_above * exp_Bc
            dD_dBc = a_c * dDc_dBc
            actions[self._nl_params["B_c"].input_name] = lambda V, c=dD_dBc: c * V

        # ∂D/∂E. Both ε_t and ε_c scale as 1/E, so each α scales as 1/E,
        # so D ∝ 1/E in its α dependence (D_t and D_c don't depend on E):
        #   ∂D/∂E = −D/E.
        if "E" in self._nl_params:
            dD_dE = -damage / E_
            actions[self._nl_params["E"].input_name] = lambda V, c=dD_dE: c * V

        # ∂D/∂ν. ∂ε_t/∂ν = (σ_t − tr(σ_t)·1) / E, so:
        #   ∂α_t/∂ν = [inner(σ_t, ε_pos) − tr(σ_t)·tr(ε_pos)] / (E · den_safe)
        # ∂D/∂ν = D_t · ∂α_t/∂ν + D_c · ∂α_c/∂ν.
        if "nu" in self._nl_params:
            da_t_dnu = (inner(sig_t, eps_pos) - tr_sig_t * tr_eps_pos) / (E_ * den_safe)
            da_c_dnu = (inner(sig_c, eps_pos) - tr_sig_c * tr_eps_pos) / (E_ * den_safe)
            dD_dnu = Dt * da_t_dnu + Dc * da_c_dnu
            actions[self._nl_params["nu"].input_name] = lambda V, c=dD_dnu: c * V

        return damage, self.apply_chain_rule(v, "damage", actions, output=damage)


__all__ = ["MazarsDamageStressAlpha"]
