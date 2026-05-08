// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/solid_mechanics/traction_separation/BiLinearMixedModeTractionSeparation.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/outer.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/where.h"

#include <cmath>

namespace neml2
{
register_NEML2_object(BiLinearMixedModeTractionSeparation);

OptionSet
BiLinearMixedModeTractionSeparation::expected_options()
{
  OptionSet options = TractionSeparation::expected_options();
  options.doc() =
      "Bilinear mixed-mode cohesive-zone traction-separation law (Camanho/Davila form) with the "
      "Benzeggagh-Kenane (BK) or power-law mixed-mode propagation criterion. Damage is "
      "irreversible. The normal compressive branch is restored through a Macaulay split so "
      "interpenetration does not soften the interface. Variable derivatives are hand-coded.";

  options.add_parameter<Scalar>("penalty_stiffness", "Elastic penalty stiffness K");
  options.add_parameter<Scalar>("normal_fracture_energy",
                                "Mode I critical energy release rate G_Ic");
  options.add_parameter<Scalar>("shear_fracture_energy",
                                "Mode II critical energy release rate G_IIc");
  options.add_parameter<Scalar>("normal_strength", "Tensile (normal) strength N");
  options.add_parameter<Scalar>("shear_strength", "Shear strength S");
  options.add_parameter<Scalar>("eta", "Mixed-mode criterion exponent (BK or power-law)");

  EnumSelection criterion({"BK", "POWER_LAW"}, "BK");
  options.add<EnumSelection>(
      "criterion", criterion, "Mixed-mode propagation criterion. Options are: " + criterion.join());
  options.add<double>("epsilon",
                      1e-16,
                      "Small regularizer added inside sqrt() and pow() bases to keep their "
                      "derivatives bounded at zero displacement jump");

  return options;
}

BiLinearMixedModeTractionSeparation::BiLinearMixedModeTractionSeparation(const OptionSet & options)
  : TractionSeparation(options),
    _d(declare_output_variable<Scalar>("damage")),
    _d_old(declare_input_variable<Scalar>(history_name(_d.name(), /*nstep=*/1))),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", true)),
    _GIc(declare_parameter<Scalar>("GIc", "normal_fracture_energy", true)),
    _GIIc(declare_parameter<Scalar>("GIIc", "shear_fracture_energy", true)),
    _N(declare_parameter<Scalar>("N", "normal_strength", true)),
    _S(declare_parameter<Scalar>("S", "shear_strength", true)),
    _eta(declare_parameter<Scalar>("eta", "eta", true)),
    _criterion(options.get<EnumSelection>("criterion")),
    _eps(options.get<double>("epsilon"))
{
}

void
BiLinearMixedModeTractionSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // ============================================================
  // Forward
  // ============================================================
  const auto dn = _delta()(0);
  const auto ds1 = _delta()(1);
  const auto ds2 = _delta()(2);

  const auto zero_s = Scalar::zeros_like(dn);
  const auto one_s = Scalar::ones_like(dn);
  const auto zero_v = Vec::zeros_like(_delta());

  // Macaulay split on the normal jump.
  // H = d(macaulay)/d(dn): 1 in opening, 0 in compression.
  const auto delta_n_pos = neml2::macaulay(dn);
  const auto delta_n_neg = dn - delta_n_pos;
  const auto H = neml2::heaviside(dn);
  const auto one_minus_H = 1.0 - H;

  // Tangential magnitude (regularized so its derivative is finite at zero).
  const auto delta_s_sq = ds1 * ds1 + ds2 * ds2;
  const auto delta_s = neml2::sqrt(delta_s_sq + _eps);

  // Mode mixity beta = delta_s / delta_n is only meaningful for opening.
  // The "safe" denominator gives finite values in the compression branch so
  // the masked-off `where` doesn't trip on division by zero.
  // All where()-conditions are detached because (1) `where` does not backpropagate
  // through its condition, so the gradient information is unused, and (2) the
  // TorchScript tracer rejects grad-tracking masks captured into the JIT graph
  // ("Cannot insert a Tensor that requires grad as a constant.").
  const auto pos_mask = (dn > 0.0).detach();
  const auto safe_delta_n_pos = neml2::where(pos_mask, delta_n_pos, one_s);
  const auto beta_open = delta_s / safe_delta_n_pos;
  const auto beta = neml2::where(pos_mask, beta_open, zero_s);
  const auto beta_sq = beta * beta;

  // Initiation displacement jump (mixed-mode in opening, pure-shear in compression).
  const auto delta_normal0 = _N / _K;
  const auto delta_shear0 = _S / _K;
  const auto delta_mixed_init_sq =
      delta_shear0 * delta_shear0 + beta_sq * delta_normal0 * delta_normal0 + _eps;
  const auto delta_mixed_init = neml2::sqrt(delta_mixed_init_sq);
  const auto delta_init_mixed =
      delta_normal0 * delta_shear0 * neml2::sqrt(1.0 + beta_sq) / delta_mixed_init;
  const auto delta_init = neml2::where(pos_mask, delta_init_mixed, delta_shear0);

  // Final (full-degradation) displacement jump per the active criterion.
  const auto Kdelta_init_mixed = _K * delta_init_mixed;
  Scalar delta_final_mixed = zero_s;
  if (_criterion == "BK")
  {
    const auto beta_sq_ratio = beta_sq / (1.0 + beta_sq);
    const auto pow_base_BK = beta_sq_ratio + _eps;
    const auto term = _GIc + (_GIIc - _GIc) * neml2::pow(pow_base_BK, _eta);
    delta_final_mixed = 2.0 / Kdelta_init_mixed * term;
  }
  else // POWER_LAW
  {
    const auto pow_base_PL = beta_sq / _GIIc + _eps;
    const auto Gc_mixed = neml2::pow(1.0 / _GIc, _eta) + neml2::pow(pow_base_PL, _eta);
    delta_final_mixed =
        (2.0 + 2.0 * beta_sq) / Kdelta_init_mixed * neml2::pow(Gc_mixed, -1.0 / _eta);
  }
  const auto delta_final_default = Scalar::full_like(dn, std::sqrt(2.0)) * 2.0 * _GIIc / _S;
  const auto delta_final = neml2::where(pos_mask, delta_final_mixed, delta_final_default);

  // Effective mixed-mode displacement jump.
  const auto delta_m = neml2::sqrt(delta_n_pos * delta_n_pos + delta_s_sq + _eps);

  // Bilinear damage (with safe denominator for the masked-off branches).
  const auto df_minus_di = delta_final - delta_init;
  const auto df_minus_di_pos_mask = (df_minus_di > 0.0).detach();
  const auto safe_df_minus_di = neml2::where(df_minus_di_pos_mask, df_minus_di, one_s);
  const auto bilinear_d = delta_final * (delta_m - delta_init) / (delta_m * safe_df_minus_di);
  const auto dm_lt_init_mask = (delta_m < delta_init).detach();
  const auto dm_lt_final_mask = (delta_m < delta_final).detach();
  const auto dm_gt_init_mask = (delta_m > delta_init).detach();
  const auto linear_mask = (dm_gt_init_mask && dm_lt_final_mask).detach();
  const auto d_trial =
      neml2::where(dm_lt_init_mask, zero_s, neml2::where(dm_lt_final_mask, bilinear_d, one_s));

  // Irreversibility: damage can only grow.
  const auto advance_mask = (d_trial > _d_old()).detach();
  const auto d = neml2::where(advance_mask, d_trial, _d_old());

  // Active/inactive split for traction.
  const auto delta_active = Vec::fill(delta_n_pos, ds1, ds2);
  const auto delta_inactive = Vec::fill(delta_n_neg, zero_s, zero_s);
  const auto T_active = _K * (1.0 - d) * delta_active;
  const auto T_inactive = _K * delta_inactive;

  if (out)
  {
    _T = T_active + T_inactive;
    _d = d;
  }

  if (!dout_din)
    return;

  // ============================================================
  // Analytical Jacobians
  // ============================================================

  // ---- d(beta)/d(delta) in the opening branch ----
  // dbeta/ddn = -delta_s / delta_n^2
  // dbeta/dds_i = ds_i / (delta_s * delta_n)
  const auto inv_dn = 1.0 / safe_delta_n_pos;
  const auto inv_ds = 1.0 / delta_s; // delta_s is bounded below by sqrt(eps) > 0
  const auto dbeta_ddn_open = -delta_s * inv_dn * inv_dn;
  const auto dbeta_dds1_open = ds1 * inv_ds * inv_dn;
  const auto dbeta_dds2_open = ds2 * inv_ds * inv_dn;
  const auto dbeta_ddelta_open = Vec::fill(dbeta_ddn_open, dbeta_dds1_open, dbeta_dds2_open);

  // ---- d(delta_init)/d(delta) in the opening branch ----
  // d(delta_init)/d(beta) = delta_init * beta * [1/(1+beta^2) - delta_n0^2/delta_mixed^2]
  const auto ddelta_init_dbeta =
      delta_init_mixed * beta *
      (1.0 / (1.0 + beta_sq) - delta_normal0 * delta_normal0 / delta_mixed_init_sq);
  const auto ddelta_init_ddelta_open = ddelta_init_dbeta * dbeta_ddelta_open;
  const auto ddelta_init_ddelta = neml2::where(pos_mask, ddelta_init_ddelta_open, zero_v);

  // ---- d(delta_final)/d(delta) in the opening branch ----
  // Both criteria share the form delta_final = h(delta_init, beta), so
  // d(delta_final)/d(delta) = (d/dinit)(delta_final) * d(init)/d(delta) +
  //                            (d/dbeta)(delta_final) * d(beta)/d(delta).
  Vec ddelta_final_ddelta_open = zero_v;
  if (_criterion == "BK")
  {
    const auto beta_sq_ratio = beta_sq / (1.0 + beta_sq);
    const auto pow_base_BK = beta_sq_ratio + _eps;
    // d(delta_final)/d(delta_init) = -delta_final / delta_init
    const auto ddelta_final_ddelta_init_BK = -delta_final_mixed / delta_init_mixed;
    // d(beta_sq_ratio)/d(beta) = 2*beta / (1+beta^2)^2
    const auto one_plus_beta_sq = 1.0 + beta_sq;
    const auto dbeta_sq_ratio_dbeta = 2.0 * beta / (one_plus_beta_sq * one_plus_beta_sq);
    // d(delta_final)/d(beta) = (2/(K*delta_init)) * (GIIc - GIc) * eta * (r+eps)^(eta-1) * dr/dbeta
    const auto ddelta_final_dbeta_BK = (2.0 / Kdelta_init_mixed) * (_GIIc - _GIc) * _eta *
                                       neml2::pow(pow_base_BK, _eta - 1.0) * dbeta_sq_ratio_dbeta;
    ddelta_final_ddelta_open = ddelta_final_ddelta_init_BK * ddelta_init_ddelta_open +
                               ddelta_final_dbeta_BK * dbeta_ddelta_open;
  }
  else // POWER_LAW
  {
    const auto pow_base_PL = beta_sq / _GIIc + _eps;
    const auto Gc_mixed = neml2::pow(1.0 / _GIc, _eta) + neml2::pow(pow_base_PL, _eta);
    const auto Gc_term = neml2::pow(Gc_mixed, -1.0 / _eta);
    const auto prefactor = (2.0 + 2.0 * beta_sq) / Kdelta_init_mixed;
    const auto dprefactor_dbeta = 4.0 * beta / Kdelta_init_mixed;
    const auto ddelta_final_ddelta_init_PL = -prefactor / delta_init_mixed * Gc_term;
    // dGc_mixed/dbeta = eta * (beta^2/GIIc + eps)^(eta-1) * (2*beta/GIIc)
    const auto dGc_mixed_dbeta = _eta * neml2::pow(pow_base_PL, _eta - 1.0) * (2.0 * beta / _GIIc);
    // dGc_term/dbeta = (-1/eta) * Gc_mixed^(-1/eta - 1) * dGc_mixed/dbeta
    const auto dGc_term_dbeta =
        (-1.0 / _eta) * neml2::pow(Gc_mixed, -1.0 / _eta - 1.0) * dGc_mixed_dbeta;
    const auto ddelta_final_dbeta_PL = dprefactor_dbeta * Gc_term + prefactor * dGc_term_dbeta;
    ddelta_final_ddelta_open = ddelta_final_ddelta_init_PL * ddelta_init_ddelta_open +
                               ddelta_final_dbeta_PL * dbeta_ddelta_open;
  }
  const auto ddelta_final_ddelta = neml2::where(pos_mask, ddelta_final_ddelta_open, zero_v);

  // ---- d(delta_m)/d(delta) ----
  // delta_m = sqrt(delta_n_pos^2 + delta_s_sq + eps).
  // The normal component is delta_n_pos / delta_m, which already vanishes in compression.
  const auto inv_dm = 1.0 / delta_m;
  const auto ddelta_m_ddelta = Vec::fill(delta_n_pos * inv_dm, ds1 * inv_dm, ds2 * inv_dm);

  // ---- d(d_trial)/d(delta) in the linear damage regime ----
  // d_trial = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
  // Partials (derivation in design/.../BiLinearMixedModeTraction_spec_simple.md):
  //   d/d(delta_m)     =  delta_final * delta_init / (delta_m^2 * (delta_final - delta_init))
  //   d/d(delta_init)  =  delta_final * (delta_m - delta_final) /
  //                       (delta_m * (delta_final - delta_init)^2)
  //   d/d(delta_final) = -delta_init * (delta_m - delta_init) /
  //                       (delta_m * (delta_final - delta_init)^2)
  const auto inv_dm_sq = inv_dm * inv_dm;
  const auto inv_df_minus_di = 1.0 / safe_df_minus_di;
  const auto inv_df_minus_di_sq = inv_df_minus_di * inv_df_minus_di;
  const auto dd_trial_ddm = delta_final * delta_init * inv_dm_sq * inv_df_minus_di;
  const auto dd_trial_ddinit = delta_final * (delta_m - delta_final) * inv_dm * inv_df_minus_di_sq;
  const auto dd_trial_ddfinal = -delta_init * (delta_m - delta_init) * inv_dm * inv_df_minus_di_sq;
  const auto dd_trial_ddelta_linear = dd_trial_ddm * ddelta_m_ddelta +
                                      dd_trial_ddinit * ddelta_init_ddelta +
                                      dd_trial_ddfinal * ddelta_final_ddelta;
  const auto dd_trial_ddelta = neml2::where(linear_mask, dd_trial_ddelta_linear, zero_v);

  // ---- Irreversibility: d/dd_old picks up the frozen branch ----
  const auto dd_ddelta = neml2::where(advance_mask, dd_trial_ddelta, zero_v);
  const auto dd_dd_old = neml2::where(advance_mask, zero_s, one_s);

  // ---- d(T)/d(delta) ----
  // T = K*(1-d)*delta_active + K*delta_inactive
  //   d(delta_active)/d(delta)   = diag(H, 1, 1)
  //   d(delta_inactive)/d(delta) = diag(1-H, 0, 0)
  // dT/d(delta) = K*(1-d) * diag(H, 1, 1) + K * diag(1-H, 0, 0)
  //               - K * outer(delta_active, dd/d(delta))
  const auto active_jac = R2::fill(H, one_s, one_s);
  const auto inactive_jac = R2::fill(one_minus_H, zero_s, zero_s);
  _T.d(_delta) =
      _K * (1.0 - d) * active_jac + _K * inactive_jac - _K * neml2::outer(delta_active, dd_ddelta);

  // ---- d(T)/d(d_old) = -K * delta_active * d(d)/d(d_old) ----
  _T.d(_d_old) = -_K * delta_active * dd_dd_old;

  _d.d(_delta) = dd_ddelta;
  _d.d(_d_old) = dd_dd_old;
}
} // namespace neml2
