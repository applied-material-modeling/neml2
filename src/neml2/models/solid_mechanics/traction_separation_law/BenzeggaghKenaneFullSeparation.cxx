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

#include "neml2/models/solid_mechanics/traction_separation_law/BenzeggaghKenaneFullSeparation.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(BenzeggaghKenaneFullSeparation);

OptionSet
BenzeggaghKenaneFullSeparation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Mixed-mode full (failure) separation under the Benzeggagh-Kenane criterion. "
                  "Opening: \\f$ \\delta_f = (2/(K \\delta_c))(G_{Ic} + (G_{IIc}-G_{Ic}) "
                  "(\\beta^2/(1+\\beta^2))^\\eta) \\f$, where \\f$ \\delta_c \\f$ is the critical "
                  "separation supplied by an upstream `CamanhoDavilaCriticalSeparation`. "
                  "Compression: \\f$ \\delta_f = 2 G_{IIc}/S \\f$ (pure-shear closed form).";

  options.add_input("normal_separation",
                    "Normal separation (typically the Macaulay-positive part of the normal jump; "
                    "used only to determine the opening / compression branch)");
  options.add_output("full_separation", "Full (failure) separation");
  options.add_parameter<Scalar>(
      "mode_mixity",
      "Mode-mixity ratio. May be wired to an upstream `ModeMixity` (nonlinear-capable).");
  options.add_parameter<Scalar>("critical_separation",
                                "Critical (damage-onset) separation. May be wired to an upstream "
                                "`CamanhoDavilaCriticalSeparation` (nonlinear-capable).");
  options.add_parameter<Scalar>("penalty_stiffness", "Penalty stiffness");
  options.add_parameter<Scalar>("mode_I_fracture_toughness", "Mode I critical energy release rate");
  options.add_parameter<Scalar>("mode_II_fracture_toughness",
                                "Mode II critical energy release rate");
  options.add_parameter<Scalar>("shear_strength", "Shear strength (used in compression branch)");
  options.add_parameter<Scalar>("eta", "BK exponent");

  return options;
}

BenzeggaghKenaneFullSeparation::BenzeggaghKenaneFullSeparation(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("full_separation")),
    _dn(declare_input_variable<Scalar>("normal_separation")),
    _beta(declare_parameter<Scalar>("beta", "mode_mixity", /*allow_nonlinear=*/true)),
    _delta_init(declare_parameter<Scalar>("delta_c",
                                          "critical_separation",
                                          /*allow_nonlinear=*/true)),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", /*allow_nonlinear=*/false)),
    _GIc(declare_parameter<Scalar>("GIc",
                                   "mode_I_fracture_toughness",
                                   /*allow_nonlinear=*/false)),
    _GIIc(declare_parameter<Scalar>("GIIc",
                                    "mode_II_fracture_toughness",
                                    /*allow_nonlinear=*/false)),
    _S(declare_parameter<Scalar>("S", "shear_strength", /*allow_nonlinear=*/false)),
    _eta(declare_parameter<Scalar>("eta", "eta", /*allow_nonlinear=*/false))
{
}

void
BenzeggaghKenaneFullSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_dn.scalar_type());
  const auto pos_mask = (_dn() > 0.0).detach();

  // _beta and _delta_init are nonlinear-capable parameters — used here as raw Scalars (no
  // operator() like input variables).
  const auto beta_sq = _beta * _beta;
  const auto one_plus_beta_sq = 1.0 + beta_sq;
  const auto beta_sq_ratio = beta_sq / one_plus_beta_sq;
  const auto pow_base = beta_sq_ratio + eps;
  const auto term = _GIc + (_GIIc - _GIc) * neml2::pow(pow_base, _eta);
  const auto delta_final_open = 2.0 / (_K * _delta_init) * term;
  const auto delta_final_default = 2.0 * _GIIc / _S;

  if (out)
    _to = neml2::where(pos_mask, delta_final_open, Scalar(delta_final_default));

  if (dout_din)
  {
    const auto zero = Scalar::zeros_like(_dn());
    _to.d(_dn) = zero;

    // Nonlinear-parameter Jacobians: only emit when the corresponding parameter is wired to an
    // upstream variable. Lookup keys are the internal first-arg names of declare_parameter
    // (delta_c, beta). d(delta_f)/d(delta_c) = -delta_f / delta_c (opening only).
    if (const auto * const dc = nl_param("delta_c"))
    {
      const auto ddf_dinit_open = -delta_final_open / _delta_init;
      _to.d(*dc) = neml2::where(pos_mask, ddf_dinit_open, zero);
    }
    if (const auto * const b = nl_param("beta"))
    {
      // d(beta_sq_ratio)/d(beta) = 2 beta / (1+beta^2)^2
      const auto dbeta_sq_ratio_dbeta = 2.0 * _beta / (one_plus_beta_sq * one_plus_beta_sq);
      // d(delta_f)/d(beta) = (2/(K delta_c)) (GIIc-GIc) eta (r+eps)^(eta-1) dr/dbeta
      const auto ddf_dbeta_open = (2.0 / (_K * _delta_init)) * (_GIIc - _GIc) * _eta *
                                  neml2::pow(pow_base, _eta - 1.0) * dbeta_sq_ratio_dbeta;
      _to.d(*b) = neml2::where(pos_mask, ddf_dbeta_open, zero);
    }
  }
}
} // namespace neml2
