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

#include "neml2/models/solid_mechanics/traction_separation_law/BilinearTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(BilinearTraction);

OptionSet
BilinearTraction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Bilinear cohesive-zone traction with internal damage state. Computes the bilinear damage "
      "variable from \\f$ (\\delta_m, \\delta_c, \\delta_f) \\f$ (effective separation, critical "
      "/ damage-onset separation, full / failure separation), caps it for irreversibility against "
      "the previous-step value (auto-declared via `history_name`), and assembles \\f$ T_n = "
      "K(1-d)\\delta_n^+ + K\\delta_n^-,\\;T_{si} = K(1-d)\\delta_{si} \\f$. Damage is exposed "
      "as a secondary output for diagnostics; the irreversibility cap is internal to this model "
      "and does not require an external `IrreversibleScalar`.";

  options.add_input("effective_separation", "Effective separation");
  options.add_input("normal_separation",
                    "Normal separation (typically the Macaulay-positive part of the normal jump)");
  options.add_input("normal_penetration",
                    "Optional normal penetration (typically the Macaulay-negative part of the "
                    "normal jump). When supplied, K times this is added to the normal traction "
                    "as a penalty term resisting interpenetration.");
  options.add_input("tangential_separation_1", "First tangential separation");
  options.add_input("tangential_separation_2", "Second tangential separation");
  options.add_output("traction", "Traction Vec");
  options.add_output("damage", "Damage scalar (current step, irreversibility-capped)");
  options.add_parameter<Scalar>("penalty_stiffness", "Penalty stiffness");
  options.add_parameter<Scalar>("critical_separation",
                                "Critical (damage-onset) separation. May be wired to an upstream "
                                "`CamanhoDavilaCriticalSeparation` (nonlinear-capable).");
  options.add_parameter<Scalar>("full_separation",
                                "Full (failure) separation. May be wired to an upstream "
                                "`BenzeggaghKenaneFullSeparation` or `PowerLawFullSeparation` "
                                "(nonlinear-capable).");

  return options;
}

BilinearTraction::BilinearTraction(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Vec>("traction")),
    _d(declare_output_variable<Scalar>("damage")),
    _d_old(declare_input_variable<Scalar>(history_name(_d.name(), /*nstep=*/1))),
    _delta_m(declare_input_variable<Scalar>("effective_separation")),
    _delta_init(declare_parameter<Scalar>("delta_c",
                                          "critical_separation",
                                          /*allow_nonlinear=*/true)),
    _delta_final(declare_parameter<Scalar>("delta_f",
                                           "full_separation",
                                           /*allow_nonlinear=*/true)),
    _dn_sep(declare_input_variable<Scalar>("normal_separation")),
    _dn_pen(options.user_specified("normal_penetration")
                ? &declare_input_variable<Scalar>("normal_penetration")
                : nullptr),
    _ds1(declare_input_variable<Scalar>("tangential_separation_1")),
    _ds2(declare_input_variable<Scalar>("tangential_separation_2")),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", /*allow_nonlinear=*/false))
{
}

void
BilinearTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto one = Scalar::ones_like(_delta_m());
  const auto zero = Scalar::zeros_like(_delta_m());

  // -------- Bilinear damage: trial value from (delta_m, delta_c, delta_f). Note that
  // _delta_init and _delta_final are nonlinear-capable parameters declared with
  // allow_nonlinear=true, so they are referenced directly (no operator() like input variables).
  // Detached masks for the where()-conditions (TorchScript can't capture grad-tracking masks).
  const auto df_minus_di = _delta_final - _delta_init;
  const auto df_minus_di_pos = (df_minus_di > 0.0).detach();
  const auto safe_df_minus_di = neml2::where(df_minus_di_pos, df_minus_di, one);
  const auto bilinear_d =
      _delta_final * (_delta_m() - _delta_init) / (_delta_m() * safe_df_minus_di);
  const auto dm_lt_init = (_delta_m() < _delta_init).detach();
  const auto dm_lt_final = (_delta_m() < _delta_final).detach();
  const auto dm_gt_init = (_delta_m() > _delta_init).detach();
  const auto interior = (dm_gt_init && dm_lt_final).detach();
  const auto d_trial = neml2::where(dm_lt_init, zero, neml2::where(dm_lt_final, bilinear_d, one));

  // -------- Irreversibility cap: damage = max(d_trial, d_old).
  const auto advance = (d_trial > _d_old()).detach();
  const auto d = neml2::where(advance, d_trial, _d_old());

  // -------- Active / inactive traction assembly.
  const auto active_scale = _K * (1.0 - d);

  if (out)
  {
    auto T_n = active_scale * _dn_sep();
    if (_dn_pen)
      T_n = T_n + _K * (*_dn_pen)();
    const auto T_s1 = active_scale * _ds1();
    const auto T_s2 = active_scale * _ds2();
    _to = Vec::fill(T_n, T_s1, T_s2);
    _d = d;
  }

  if (!dout_din)
    return;

  // -------- Jacobians.
  // d(d_trial) partials in the interior (zero elsewhere via the `interior` mask):
  //   d/d(delta_m) =  delta_f * delta_c / (delta_m^2 (delta_f - delta_c))
  //   d/d(delta_c) =  delta_f (delta_m - delta_f) / (delta_m (delta_f - delta_c)^2)
  //   d/d(delta_f) = -delta_c (delta_m - delta_c) / (delta_m (delta_f - delta_c)^2)
  const auto inv_dm = 1.0 / _delta_m();
  const auto inv_diff = 1.0 / safe_df_minus_di;
  const auto inv_diff_sq = inv_diff * inv_diff;

  const auto dt_ddm_int = _delta_final * _delta_init * inv_dm * inv_dm * inv_diff;
  const auto dt_ddm = neml2::where(interior, dt_ddm_int, zero);

  // Irreversibility freezes partials when d_trial doesn't advance:
  const auto dd_ddm = neml2::where(advance, dt_ddm, zero);
  const auto dd_dd_old = neml2::where(advance, zero, one);

  // d(damage)/...
  _d.d(_delta_m) = dd_ddm;
  _d.d(_d_old) = dd_dd_old;

  // d(traction)/... routes through (1) the active_scale = K(1-d) factor on
  // (δ_n^sep, δ_s1, δ_s2), (2) the K factor on δ_n^pen for T_n (when supplied), and (3) the
  // per-component multipliers for direct partials.
  const auto neg_K = -_K;
  // dT/d(damage) = -K * (δ_n^sep, δ_s1, δ_s2). Then dT/d(scalar driving damage) =
  // dT/d(damage) * dd/d(scalar).
  const auto dT_dd_n = neg_K * _dn_sep();
  const auto dT_dd_s1 = neg_K * _ds1();
  const auto dT_dd_s2 = neg_K * _ds2();

  _to.d(_delta_m) = Vec::fill(dT_dd_n * dd_ddm, dT_dd_s1 * dd_ddm, dT_dd_s2 * dd_ddm);
  _to.d(_d_old) = Vec::fill(dT_dd_n * dd_dd_old, dT_dd_s1 * dd_dd_old, dT_dd_s2 * dd_dd_old);

  // Nonlinear-parameter Jacobians: only emit when the user wired them to an upstream variable
  // (allow_nonlinear=true). When supplied as literal numbers, nl_param() returns nullptr.
  // Lookup keys are the internal first-arg names of declare_parameter (delta_c, delta_f).
  if (const auto * const dc = nl_param("delta_c"))
  {
    const auto dt_dinit_int = _delta_final * (_delta_m() - _delta_final) * inv_dm * inv_diff_sq;
    const auto dt_dinit = neml2::where(interior, dt_dinit_int, zero);
    const auto dd_dinit = neml2::where(advance, dt_dinit, zero);
    _d.d(*dc) = dd_dinit;
    _to.d(*dc) = Vec::fill(dT_dd_n * dd_dinit, dT_dd_s1 * dd_dinit, dT_dd_s2 * dd_dinit);
  }
  if (const auto * const df = nl_param("delta_f"))
  {
    const auto dt_dfinal_int = -_delta_init * (_delta_m() - _delta_init) * inv_dm * inv_diff_sq;
    const auto dt_dfinal = neml2::where(interior, dt_dfinal_int, zero);
    const auto dd_dfinal = neml2::where(advance, dt_dfinal, zero);
    _d.d(*df) = dd_dfinal;
    _to.d(*df) = Vec::fill(dT_dd_n * dd_dfinal, dT_dd_s1 * dd_dfinal, dT_dd_s2 * dd_dfinal);
  }

  // Direct partials w.r.t. the per-component jumps (d held fixed there, since the cap is on
  // damage, not on the jumps themselves).
  _to.d(_dn_sep) = Vec::fill(active_scale, zero, zero);
  if (_dn_pen)
    _to.d(*_dn_pen) = Vec::fill(Scalar(_K), zero, zero);
  _to.d(_ds1) = Vec::fill(zero, active_scale, zero);
  _to.d(_ds2) = Vec::fill(zero, zero, active_scale);
}
} // namespace neml2
