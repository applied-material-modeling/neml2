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

#include "neml2/models/solid_mechanics/cohesive/BiLinearMixedModeTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/where.h"

#include <cmath>
#include <stdexcept>

namespace neml2
{
register_NEML2_object(BiLinearMixedModeTraction);

OptionSet
BiLinearMixedModeTraction::expected_options()
{
  OptionSet options = TractionSeparationModel::expected_options();
  options.doc() +=
      " following a bilinear mixed-mode damage law with irreversible damage evolution "
      "and optional viscous regularization (Camanho & Davila, NASA/TM-2002-211737).";

  options.set_parameter<TensorName<Scalar>>("penalty_stiffness");
  options.set("penalty_stiffness").doc() = "Penalty elastic stiffness \\f$ K \\f$";

  options.set_parameter<TensorName<Scalar>>("mode_I_critical_fracture_energy");
  options.set("mode_I_critical_fracture_energy").doc() =
      "Mode I critical energy release rate \\f$ G_{\\mathrm{Ic}} \\f$";

  options.set_parameter<TensorName<Scalar>>("mode_II_critical_fracture_energy");
  options.set("mode_II_critical_fracture_energy").doc() =
      "Mode II critical energy release rate \\f$ G_{\\mathrm{IIc}} \\f$";

  options.set_parameter<TensorName<Scalar>>("normal_strength");
  options.set("normal_strength").doc() = "Tensile (normal) interface strength \\f$ N \\f$";

  options.set_parameter<TensorName<Scalar>>("shear_strength");
  options.set("shear_strength").doc() = "Shear interface strength \\f$ S \\f$";

  options.set_parameter<TensorName<Scalar>>("mixed_mode_exponent");
  options.set("mixed_mode_exponent").doc() =
      "Mixed-mode propagation exponent \\f$ \\eta \\f$ for BK or power-law criterion";

  options.set_parameter<TensorName<Scalar>>("viscosity");
  options.set("viscosity").doc() =
      "Viscous regularization coefficient (0 disables regularization)";

  options.set_input("damage_old") = VariableName(OLD_STATE, "damage");
  options.set("damage_old").doc() = "Damage variable from the previous time step";

  options.set_input("displacement_jump_old") = VariableName(OLD_FORCES, "displacement_jump");
  options.set("displacement_jump_old").doc() =
      "Displacement jump from previous step, used when lag_mode_mixity or lag_displacement_jump "
      "is enabled";

  options.set_input("time") = VariableName(FORCES, "t");
  options.set("time").doc() = "Current time";

  options.set_input("time_old") = VariableName(OLD_FORCES, "t");
  options.set("time_old").doc() = "Time at the start of the step, used to compute dt";

  options.set_output("damage") = VariableName(STATE, "damage");
  options.set("damage").doc() =
      "Damage variable after irreversibility enforcement and viscous regularization";

  options.set<bool>("lag_mode_mixity") = true;
  options.set("lag_mode_mixity").doc() =
      "Use displacement jump from previous step when computing mode-mixity ratio beta, "
      "delta_init, and delta_final (default true)";

  options.set<bool>("lag_displacement_jump") = false;
  options.set("lag_displacement_jump").doc() =
      "Use displacement jump from previous step when computing effective mixed-mode "
      "displacement jump delta_m (default false)";

  options.set<std::string>("criterion") = "BK";
  options.set("criterion").doc() =
      "Mixed-mode propagation criterion: \"BK\" (Benzeggagh-Kenane) or \"POWER_LAW\"";

  return options;
}

BiLinearMixedModeTraction::BiLinearMixedModeTraction(const OptionSet & options)
  : TractionSeparationModel(options),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness")),
    _GI_c(declare_parameter<Scalar>("GI_c", "mode_I_critical_fracture_energy")),
    _GII_c(declare_parameter<Scalar>("GII_c", "mode_II_critical_fracture_energy")),
    _N(declare_parameter<Scalar>("N", "normal_strength")),
    _S(declare_parameter<Scalar>("S", "shear_strength")),
    _eta(declare_parameter<Scalar>("eta", "mixed_mode_exponent")),
    _viscosity(declare_parameter<Scalar>("viscosity", "viscosity")),
    _d_old(declare_input_variable<Scalar>("damage_old")),
    _delta_old(declare_input_variable<Vec>("displacement_jump_old")),
    _t_old(declare_input_variable<Scalar>("time_old")),
    _t(declare_input_variable<Scalar>("time")),
    _d(declare_output_variable<Scalar>("damage")),
    _lag_mode_mixity(options.get<bool>("lag_mode_mixity")),
    _lag_disp_jump(options.get<bool>("lag_displacement_jump")),
    _criterion(options.get<std::string>("criterion"))
{
  if (_criterion != "BK" && _criterion != "POWER_LAW")
    throw std::invalid_argument("BiLinearMixedModeTraction: criterion must be 'BK' or "
                                "'POWER_LAW', got '" +
                                _criterion + "'");
}

void
BiLinearMixedModeTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto delta_cur = _delta();
  const auto dn_cur = delta_cur(0);
  const auto ds1_cur = delta_cur(1);
  const auto ds2_cur = delta_cur(2);
  const auto d_old = _d_old();
  const auto dt = _t() - _t_old();

  // ---------------------------------------------------------------
  // Step 1: Mode-mixity ratio beta = delta_s / delta_n  (or lagged)
  // ---------------------------------------------------------------
  const auto delta_mix = _lag_mode_mixity ? _delta_old() : delta_cur;
  const auto dn_m = delta_mix(0);
  const auto ds1_m = delta_mix(1);
  const auto ds2_m = delta_mix(2);

  const auto delta_s = neml2::sqrt(ds1_m * ds1_m + ds2_m * ds2_m);
  const auto zero = Scalar::zeros_like(dn_m);
  const auto one = Scalar::ones_like(dn_m);
  const auto normal_open = dn_m > zero;

  // beta: mode-mixity ratio; zero if no normal opening
  const auto safe_dn_m = where(normal_open, dn_m, one); // avoid division by zero
  const auto beta = where(normal_open, delta_s / safe_dn_m, zero);
  const auto beta_sq = beta * beta;
  const auto one_plus_beta_sq = one + beta_sq;

  // Gradient of beta w.r.t. current delta (only needed when not lagged)
  // dbeta/ddn  = -delta_s / dn^2
  // dbeta/dds1 = ds1 / (delta_s * dn)  (when delta_s > 0)
  // dbeta/dds2 = ds2 / (delta_s * dn)
  Vec dbeta_ddelta;
  if (dout_din && !_lag_mode_mixity && _delta.is_dependent())
  {
    const auto safe_delta_s = where(delta_s > zero, delta_s, one);
    const auto safe_dn2 = where(normal_open, dn_m * dn_m, one);
    const auto nonzero_s = delta_s > zero;
    const auto dbeta_ddn = where(normal_open, -delta_s / safe_dn2, zero);
    const auto dbeta_dds1 =
        where(normal_open && nonzero_s, ds1_m / (safe_delta_s * safe_dn_m), zero);
    const auto dbeta_dds2 =
        where(normal_open && nonzero_s, ds2_m / (safe_delta_s * safe_dn_m), zero);
    dbeta_ddelta = Vec::fill(dbeta_ddn, dbeta_dds1, dbeta_dds2);
  }
  else
  {
    dbeta_ddelta = Vec::fill(zero, zero, zero);
  }

  // ---------------------------------------------------------------
  // Step 2: Damage-initiation displacement jump delta_init
  // ---------------------------------------------------------------
  const auto delta_n0 = _N / _K; // N/K
  const auto delta_s0 = _S / _K; // S/K

  // When normal opening: delta_init = delta_n0*delta_s0*sqrt(1+beta^2)/delta_mixed
  // where delta_mixed = sqrt(delta_s0^2 + (beta*delta_n0)^2)
  const auto delta_mixed_sq = delta_s0 * delta_s0 + beta_sq * delta_n0 * delta_n0;
  const auto delta_mixed = neml2::sqrt(delta_mixed_sq);
  const auto safe_delta_mixed = where(delta_mixed > zero, delta_mixed, one);
  const auto delta_init_open =
      delta_n0 * delta_s0 * neml2::sqrt(one_plus_beta_sq) / safe_delta_mixed;
  const auto delta_init = where(normal_open, delta_init_open, delta_s0);

  // ddelta_init/dbeta (analytic, for chain rule)
  // ddelta_init_open/dbeta = delta_init_open * beta * (1/(1+beta^2) - delta_n0^2/delta_mixed^2)
  Scalar ddelta_init_dbeta;
  if (dout_din && !_lag_mode_mixity)
    ddelta_init_dbeta = where(normal_open,
                              delta_init_open * beta *
                                  (one / one_plus_beta_sq -
                                   delta_n0 * delta_n0 / (safe_delta_mixed * safe_delta_mixed)),
                              zero);
  else
    ddelta_init_dbeta = zero;

  // ---------------------------------------------------------------
  // Step 3: Full-degradation displacement jump delta_final
  // ---------------------------------------------------------------
  // Default (pure shear path): delta_final = sqrt(2) * 2 * GII_c / S
  const auto sqrt2 = Scalar::full(std::sqrt(2.0), _GII_c.options());
  const auto delta_final_default = sqrt2 * 2 * _GII_c / _S;

  // Computed from fracture criterion when normal opening is present
  Scalar delta_final_open;
  Scalar ddelta_final_dbeta;
  Scalar ddelta_final_ddelta_init;

  if (_criterion == "BK")
  {
    const auto beta_sq_ratio = where(one_plus_beta_sq > zero, beta_sq / one_plus_beta_sq, zero);
    const auto mix_factor = neml2::pow(beta_sq_ratio, _eta);
    const auto safe_delta_init = where(delta_init > zero, delta_init, one);
    delta_final_open =
        2 / (_K * safe_delta_init) * (_GI_c + (_GII_c - _GI_c) * mix_factor);

    if (dout_din && !_lag_mode_mixity)
    {
      ddelta_final_ddelta_init = where(normal_open, -delta_final_open / safe_delta_init, zero);
      const auto dbeta_sq_ratio_dbeta =
          where(one_plus_beta_sq > zero, 2 * beta / (one_plus_beta_sq * one_plus_beta_sq), zero);
      const auto dmix_dbeta =
          where(mix_factor > zero, _eta * neml2::pow(beta_sq_ratio, _eta - one) *
                                       dbeta_sq_ratio_dbeta,
                zero);
      ddelta_final_dbeta = where(normal_open,
                                 2 / (_K * safe_delta_init) * (_GII_c - _GI_c) * dmix_dbeta,
                                 zero);
    }
    else
    {
      ddelta_final_dbeta = zero;
      ddelta_final_ddelta_init = zero;
    }
  }
  else // POWER_LAW
  {
    const auto inv_GIc_eta = neml2::pow(_GI_c, -_eta);
    const auto safe_GIIc = where(_GII_c > zero, _GII_c, one);
    const auto b2_over_GIIc_eta = neml2::pow(beta_sq / safe_GIIc, _eta);
    const auto Gc_mixed = inv_GIc_eta + b2_over_GIIc_eta;
    const auto safe_Gc = where(Gc_mixed > zero, Gc_mixed, one);
    const auto safe_delta_init = where(delta_init > zero, delta_init, one);
    const auto Gc_neg_inv_eta = neml2::pow(safe_Gc, -one / _eta);
    delta_final_open =
        (2 + 2 * beta_sq) / (_K * safe_delta_init) * Gc_neg_inv_eta;

    if (dout_din && !_lag_mode_mixity)
    {
      ddelta_final_ddelta_init = where(normal_open, -delta_final_open / safe_delta_init, zero);
      const auto dGc_dbeta =
          where(Gc_mixed > zero,
                _eta * neml2::pow(beta_sq / safe_GIIc, _eta - one) * (2 * beta / safe_GIIc),
                zero);
      const auto prefactor = (2 + 2 * beta_sq) / (_K * safe_delta_init);
      const auto dprefactor_dbeta = 4 * beta / (_K * safe_delta_init);
      const auto dGc_term_dbeta =
          where(Gc_mixed > zero,
                (-one / _eta) * neml2::pow(safe_Gc, -one / _eta - one) * dGc_dbeta,
                zero);
      ddelta_final_dbeta =
          where(normal_open, dprefactor_dbeta * Gc_neg_inv_eta + prefactor * dGc_term_dbeta, zero);
    }
    else
    {
      ddelta_final_dbeta = zero;
      ddelta_final_ddelta_init = zero;
    }
  }

  const auto delta_final = where(normal_open, delta_final_open, delta_final_default);

  // Chain rule: ddelta_final/ddelta and ddelta_init/ddelta (via beta)
  Vec ddelta_final_ddelta;
  Vec ddelta_init_ddelta;
  if (dout_din && !_lag_mode_mixity)
  {
    ddelta_init_ddelta = ddelta_init_dbeta * dbeta_ddelta;
    ddelta_final_ddelta =
        ddelta_final_ddelta_init * ddelta_init_ddelta + ddelta_final_dbeta * dbeta_ddelta;
  }
  else
  {
    ddelta_init_ddelta = Vec::fill(zero, zero, zero);
    ddelta_final_ddelta = Vec::fill(zero, zero, zero);
  }

  // ---------------------------------------------------------------
  // Step 4: Effective mixed-mode displacement jump delta_m
  // ---------------------------------------------------------------
  const auto delta_dj = _lag_disp_jump ? _delta_old() : delta_cur;
  const auto dn_dj = delta_dj(0);
  const auto ds1_dj = delta_dj(1);
  const auto ds2_dj = delta_dj(2);

  // Positive normal part (Macaulay bracket)
  const auto dn_pos = macaulay(dn_dj);
  const auto delta_m = neml2::sqrt(ds1_dj * ds1_dj + ds2_dj * ds2_dj + dn_pos * dn_pos);

  // Gradient of delta_m w.r.t. current delta (only when not lagged)
  Vec ddelta_m_ddelta;
  if (dout_din && !_lag_disp_jump && _delta.is_dependent())
  {
    const auto nonzero_m = delta_m > zero;
    const auto inv_dm = where(nonzero_m, one / delta_m, zero);
    const auto H_n = heaviside(dn_dj);
    const auto ddm_ddn = dn_pos * H_n * inv_dm;
    const auto ddm_dds1 = ds1_dj * inv_dm;
    const auto ddm_dds2 = ds2_dj * inv_dm;
    ddelta_m_ddelta = Vec::fill(ddm_ddn, ddm_dds1, ddm_dds2);
  }
  else
  {
    ddelta_m_ddelta = Vec::fill(zero, zero, zero);
  }

  // ---------------------------------------------------------------
  // Step 5: Damage
  // ---------------------------------------------------------------
  const auto safe_delta_init = where(delta_init > zero, delta_init, one);
  const auto safe_delta_final = where(delta_final > delta_init, delta_final, delta_init + one);
  const auto denom = delta_m * (safe_delta_final - safe_delta_init);
  const auto safe_denom = where(denom > zero, denom, one);
  const auto d_bilinear = safe_delta_final * (delta_m - safe_delta_init) / safe_denom;

  const auto in_elastic = delta_m <= safe_delta_init;
  const auto fully_damaged = delta_m >= safe_delta_final;

  const auto d_trial = where(in_elastic, zero, where(fully_damaged, one, d_bilinear));

  // Irreversibility: d cannot decrease
  const auto d_irreversible = where(d_trial < d_old, d_old, d_trial);

  // Viscous regularization: d_visc = (d + visc*d_old/dt) / (visc/dt + 1)
  const auto safe_dt = where(dt > zero, dt, one);
  const auto visc_over_dt = _viscosity / safe_dt;
  const auto d_visc = (d_irreversible + _viscosity * d_old / safe_dt) / (visc_over_dt + one);

  // Gradient of d_visc w.r.t. delta (Vec)
  Vec dd_visc_ddelta;
  if (dout_din && _delta.is_dependent())
  {
    // dd_bilinear/ddelta via chain rule through delta_m, delta_init, delta_final
    // dd_bilinear/ddelta_m = delta_final*delta_init / (delta_m^2 * (delta_final - delta_init))
    const auto dd_bilinear_ddm =
        safe_delta_final * safe_delta_init / (delta_m * delta_m * (safe_delta_final - safe_delta_init));
    const auto dd_bilinear_ddelta_m = dd_bilinear_ddm * ddelta_m_ddelta;
    // ddelta_init and ddelta_final contributions (zero when lagged)
    // dd_bilinear/ddelta_init = delta_final*(delta_m*(delta_final-delta_init) - (delta_m-delta_init)*delta_m) / ...
    //                         = -delta_final / ((delta_final - delta_init)) * (1 - delta_final*(delta_m-delta_init)/(delta_m*(delta_final-delta_init)))
    // Simplified: dd_bilinear/ddelta_init = -(delta_final/delta_m) / (delta_final - delta_init)
    const auto dd_bilinear_ddelta_init_scalar =
        -safe_delta_final / (safe_denom * safe_dt / safe_dt); // simplify; recompute below
    // Actually just use the clean form:
    // d_bilinear = df * (dm - di) / (dm * (df - di))
    // dd/ddi = df * (-dm) / (dm*(df-di))  - df*(dm-di)*(-dm)/(dm*(df-di))^2
    //        = df/(df-di) * [-1/dm + (dm-di)/(dm*(df-di))]
    //        = df/(df-di) * [-(df-di) + (dm-di)] / (dm*(df-di))
    //        = df * [-df + dm] / (dm*(df-di)^2)   ... this is getting messy

    // Use the simpler quotient-rule result directly:
    // Let A = df*(dm - di), B = dm*(df - di)
    // d_bilinear = A/B
    // dA/ddi = -df, dA/ddf = dm - di
    // dB/ddi = 0,   dB/ddf = dm
    // d(A/B)/ddi = (dA/ddi * B - A * dB/ddi) / B^2 = -df / (dm*(df-di)) - 0 ... wait
    // actually dB/ddi = -dm, not 0:
    // B = dm*(df - di), dB/ddi = -dm
    // d(A/B)/ddi = (-df * B - A * (-dm)) / B^2
    //            = (-df * dm*(df-di) + df*(dm-di)*dm) / (dm*(df-di))^2
    //            = df * dm * (-(df-di) + (dm-di)) / (dm*(df-di))^2
    //            = df * (dm - df) / (dm * (df-di)^2)
    const auto df_minus_di = safe_delta_final - safe_delta_init;
    const auto dd_bilinear_ddelta_init_s =
        safe_delta_final * (delta_m - safe_delta_final) / (delta_m * df_minus_di * df_minus_di);
    // d(A/B)/ddf = ((dm-di)*B - A*dm) / B^2
    //           = ((dm-di)*dm*(df-di) - df*(dm-di)*dm) / (dm*(df-di))^2
    //           = (dm-di)*dm*(-(df)) / (dm*(df-di))^2  ... wait
    //           = dm*(dm-di)*(df-di-df) / (dm*(df-di))^2
    //           = -(dm-di)*di / (dm*(df-di)^2)
    const auto dd_bilinear_ddelta_final_s =
        -(delta_m - safe_delta_init) * safe_delta_init / (delta_m * df_minus_di * df_minus_di);

    const auto dd_bilinear_ddelta_contrib =
        dd_bilinear_ddelta_m + dd_bilinear_ddelta_init_s * ddelta_init_ddelta +
        dd_bilinear_ddelta_final_s * ddelta_final_ddelta;

    // Zero out in elastic and fully-damaged regimes
    const auto dd_trial_ddelta =
        where(in_elastic || fully_damaged, Vec::fill(zero, zero, zero), dd_bilinear_ddelta_contrib);

    // Zero out when irreversibility clamps (d_trial < d_old)
    const auto dd_irr_ddelta = where(d_trial < d_old, Vec::fill(zero, zero, zero), dd_trial_ddelta);

    // Viscous scaling
    dd_visc_ddelta = dd_irr_ddelta / (visc_over_dt + one);
  }
  else
  {
    dd_visc_ddelta = Vec::fill(zero, zero, zero);
  }

  // ---------------------------------------------------------------
  // Step 6: Traction
  // ---------------------------------------------------------------
  if (out)
  {
    _d = d_visc;

    // T = (1-d)*K*(dn_pos_cur, ds1_cur, ds2_cur) + K*(dn_neg_cur, 0, 0)
    const auto H_n_cur = heaviside(dn_cur);
    const auto dn_pos_cur = H_n_cur * dn_cur;
    const auto dn_neg_cur = dn_cur - dn_pos_cur;
    const auto one_m_d = one - d_visc;

    _traction = Vec::fill(one_m_d * _K * dn_pos_cur + _K * dn_neg_cur,
                          one_m_d * _K * ds1_cur,
                          one_m_d * _K * ds2_cur);
  }

  // ---------------------------------------------------------------
  // Step 7: Traction Jacobian
  // ---------------------------------------------------------------
  if (dout_din && _delta.is_dependent())
  {
    _d.d(_delta) = dd_visc_ddelta;

    const auto H_n_cur = heaviside(dn_cur);
    const auto dn_pos_cur = H_n_cur * dn_cur;

    // dT_n/d(dn, ds1, ds2):
    //   T_n = (1-d)*K*dn_pos + K*dn_neg = K*dn - K*d*dn_pos
    //   dT_n/ddn  = K*(1 - d*H_n) - K*dn_pos*dd/ddn
    //   dT_n/dds1 = -K*dn_pos * dd/dds1
    //   dT_n/dds2 = -K*dn_pos * dd/dds2
    const auto dTn_ddn =
        _K * (one - d_visc * H_n_cur) - _K * dn_pos_cur * dd_visc_ddelta(0);
    const auto dTn_dds1 = -_K * dn_pos_cur * dd_visc_ddelta(1);
    const auto dTn_dds2 = -_K * dn_pos_cur * dd_visc_ddelta(2);

    // dT_s1/d(dn, ds1, ds2):
    //   T_s1 = (1-d)*K*ds1_cur
    //   dT_s1/ddn  = -K*ds1_cur * dd/ddn
    //   dT_s1/dds1 = (1-d)*K - K*ds1_cur * dd/dds1
    //   dT_s1/dds2 = -K*ds1_cur * dd/dds2
    const auto dTs1_ddn = -_K * ds1_cur * dd_visc_ddelta(0);
    const auto dTs1_dds1 = (one - d_visc) * _K - _K * ds1_cur * dd_visc_ddelta(1);
    const auto dTs1_dds2 = -_K * ds1_cur * dd_visc_ddelta(2);

    // dT_s2/d(dn, ds1, ds2):
    const auto dTs2_ddn = -_K * ds2_cur * dd_visc_ddelta(0);
    const auto dTs2_dds1 = -_K * ds2_cur * dd_visc_ddelta(1);
    const auto dTs2_dds2 = (one - d_visc) * _K - _K * ds2_cur * dd_visc_ddelta(2);

    // Row-major 3x3 Jacobian
    _traction.d(_delta) =
        R2::fill(dTn_ddn, dTn_dds1, dTn_dds2, dTs1_ddn, dTs1_dds1, dTs1_dds2, dTs2_ddn,
                 dTs2_dds1, dTs2_dds2);
  }
}
} // namespace neml2
