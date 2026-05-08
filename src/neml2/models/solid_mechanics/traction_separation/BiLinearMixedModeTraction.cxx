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

#include "neml2/models/solid_mechanics/traction_separation/BiLinearMixedModeTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/where.h"

#include <cmath>

namespace neml2
{
register_NEML2_object(BiLinearMixedModeTraction);

OptionSet
BiLinearMixedModeTraction::expected_options()
{
  OptionSet options = TractionSeparation::expected_options();
  options.doc() =
      "Bilinear mixed-mode cohesive-zone traction-separation law (Camanho/Davila form) using the "
      "BK propagation criterion. The mode mixity \\f$ \\beta \\f$ and the damage-initiation / "
      "full-degradation jumps are computed from the previous-step displacement jump "
      "(lag_mode_mixity=true). The current-step effective scalar separation uses a smooth "
      "Macaulay bracket on the normal jump. Damage is irreversible: "
      "\\f$ d = \\max(d_{trial}, d_{old}) \\f$. Derivatives are obtained by automatic "
      "differentiation through the smoothed forward operator.";

  options.add_parameter<Scalar>("penalty_stiffness", "Elastic penalty stiffness K");
  options.add_parameter<Scalar>("normal_fracture_energy", "Mode I critical energy release rate G_Ic");
  options.add_parameter<Scalar>("shear_fracture_energy",
                                "Mode II critical energy release rate G_IIc");
  options.add_parameter<Scalar>("normal_strength", "Tensile (normal) strength N");
  options.add_parameter<Scalar>("shear_strength", "Shear strength S");
  options.add_parameter<Scalar>("eta", "BK power-law exponent");

  options.add<double>("alpha",
                      1e-4,
                      "Smoothing parameter for the regularized Heaviside used in the Macaulay "
                      "bracket and for guarding divisions / square roots");

  // The model relies on automatic differentiation, which is incompatible with JIT tracing.
  options.set<bool>("jit", false);

  return options;
}

BiLinearMixedModeTraction::BiLinearMixedModeTraction(const OptionSet & options)
  : TractionSeparation(options),
    _d(declare_output_variable<Scalar>("damage")),
    _d_old(declare_input_variable<Scalar>(history_name(_d.name(), /*nstep=*/1))),
    _delta_old(declare_input_variable<Vec>(history_name(_delta.name(), /*nstep=*/1))),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", true)),
    _GIc(declare_parameter<Scalar>("GIc", "normal_fracture_energy", true)),
    _GIIc(declare_parameter<Scalar>("GIIc", "shear_fracture_energy", true)),
    _N(declare_parameter<Scalar>("N", "normal_strength", true)),
    _S(declare_parameter<Scalar>("S", "shear_strength", true)),
    _eta(declare_parameter<Scalar>("eta", "eta", true)),
    _alpha(options.get<double>("alpha"))
{
}

void
BiLinearMixedModeTraction::request_AD()
{
  _T.request_AD({&_delta, &_delta_old, &_d_old});
  _d.request_AD({&_delta, &_delta_old, &_d_old});
}

void
BiLinearMixedModeTraction::set_value(bool out, bool /*dout_din*/, bool /*d2out_din2*/)
{
  if (!out)
    return;

  const auto alpha2 = _alpha * _alpha;
  const auto zero = Scalar::zeros_like(_K);
  const auto one_s = Scalar::ones_like(_K);

  // Component access
  const auto delta_n = _delta()(0);
  const auto delta_t1 = _delta()(1);
  const auto delta_t2 = _delta()(2);
  const auto delta_n_o = _delta_old()(0);
  const auto delta_t1_o = _delta_old()(1);
  const auto delta_t2_o = _delta_old()(2);

  // Lagged mode mixity from previous-step jump
  const auto delta_s_old =
      neml2::sqrt(delta_t1_o * delta_t1_o + delta_t2_o * delta_t2_o + alpha2);
  const auto positive_dn_o = delta_n_o > zero;
  const auto beta = neml2::where(positive_dn_o, delta_s_old / (delta_n_o + _alpha), zero);
  const auto beta_sq = beta * beta;
  const auto one_plus_beta_sq = one_s + beta_sq;

  // Damage-initiation displacement jump
  const auto delta_n0 = _N / _K;
  const auto delta_s0 = _S / _K;
  const auto delta_mixed =
      neml2::sqrt(delta_s0 * delta_s0 + beta_sq * delta_n0 * delta_n0 + alpha2);
  const auto delta_init_open =
      delta_n0 * delta_s0 * neml2::sqrt(one_plus_beta_sq) / delta_mixed;
  const auto delta_init = neml2::where(positive_dn_o, delta_init_open, delta_s0);

  // Full-degradation displacement jump (BK criterion)
  // Add alpha2 inside the pow base to keep the derivative finite when beta_sq_ratio == 0
  const auto beta_sq_ratio = beta_sq / one_plus_beta_sq + alpha2;
  const auto delta_final_open =
      (2.0 / (_K * delta_init)) * (_GIc + (_GIIc - _GIc) * neml2::pow(beta_sq_ratio, _eta));
  const auto delta_final_default = std::sqrt(2.0) * 2.0 * _GIIc / _S;
  const auto delta_final = neml2::where(positive_dn_o, delta_final_open, delta_final_default);

  // Smooth Macaulay split on the current normal jump
  // H(x; alpha) = 0.5 * (1 + x / sqrt(x^2 + alpha^2))
  const auto sqrt_dn2_a2 = neml2::sqrt(delta_n * delta_n + alpha2);
  const auto H = 0.5 * (one_s + delta_n / sqrt_dn2_a2);
  const auto delta_n_pos = H * delta_n;
  const auto delta_n_neg = delta_n - delta_n_pos;

  // Current-step effective scalar separation
  const auto delta_m = neml2::sqrt(delta_t1 * delta_t1 + delta_t2 * delta_t2
                                   + delta_n_pos * delta_n_pos + alpha2);

  // Bilinear damage in [0, 1] with smooth clamping by where()
  const auto d_open = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init));
  const auto d_low = neml2::where(delta_m > delta_init, d_open, zero);
  const auto d_trial = neml2::where(delta_m > delta_final, one_s, d_low);

  // Irreversibility: damage is non-decreasing
  const auto d = neml2::where(d_trial > _d_old(), d_trial, _d_old());

  // Traction with normal Macaulay split: damaged on opening + tangential, elastic on compression
  const auto factor = (one_s - d) * _K;
  const auto T_n = factor * delta_n_pos + _K * delta_n_neg;
  const auto T_t1 = factor * delta_t1;
  const auto T_t2 = factor * delta_t2;

  _T = Vec::fill(T_n, T_t1, T_t2);
  _d = d;
}
} // namespace neml2
