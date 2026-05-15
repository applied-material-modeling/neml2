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

#include "neml2/models/solid_mechanics/traction_separation_law/BKCriterion.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(BKCriterion);

OptionSet
BKCriterion::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Benzeggagh-Kenane (BK) mixed-mode propagation criterion: full-degradation jump. "
                  "Opening: \\f$ \\delta_\\text{final} = (2/(K \\delta_\\text{init})) (G_{Ic} + "
                  "(G_{IIc}-G_{Ic})(\\beta^2/(1+\\beta^2))^\\eta) \\f$. Compression: "
                  "\\f$ \\delta_\\text{final} = 2 G_{IIc}/S \\f$ (pure-shear closed form).";

  options.add_input("mixity", "Mode-mixity ratio \\f$ \\beta \\f$");
  options.add_input("initiation", "Initiation displacement jump \\f$ \\delta_\\text{init} \\f$");
  options.add_input("normal",
                    "Macaulay-positive normal jump (only used to determine the opening / "
                    "compression branch)");
  options.add_output("to", "Full-degradation displacement jump \\f$ \\delta_\\text{final} \\f$");
  options.add_parameter<Scalar>("penalty_stiffness", "Penalty stiffness K");
  options.add_parameter<Scalar>("normal_fracture_toughness",
                                "Mode I critical energy release rate G_Ic");
  options.add_parameter<Scalar>("shear_fracture_toughness",
                                "Mode II critical energy release rate G_IIc");
  options.add_parameter<Scalar>("shear_strength", "Shear strength S (used in compression branch)");
  options.add_parameter<Scalar>("eta", "BK exponent");

  return options;
}

BKCriterion::BKCriterion(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to")),
    _beta(declare_input_variable<Scalar>("mixity")),
    _delta_init(declare_input_variable<Scalar>("initiation")),
    _dn_pos(declare_input_variable<Scalar>("normal")),
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", false)),
    _GIc(declare_parameter<Scalar>("GIc", "normal_fracture_toughness", false)),
    _GIIc(declare_parameter<Scalar>("GIIc", "shear_fracture_toughness", false)),
    _S(declare_parameter<Scalar>("S", "shear_strength", false)),
    _eta(declare_parameter<Scalar>("eta", "eta", false))
{
}

void
BKCriterion::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_beta.scalar_type());
  const auto pos_mask = (_dn_pos() > 0.0).detach();

  const auto beta_sq = _beta() * _beta();
  const auto one_plus_beta_sq = 1.0 + beta_sq;
  const auto beta_sq_ratio = beta_sq / one_plus_beta_sq;
  const auto pow_base = beta_sq_ratio + eps;
  const auto term = _GIc + (_GIIc - _GIc) * neml2::pow(pow_base, _eta);
  const auto delta_final_open = 2.0 / (_K * _delta_init()) * term;
  const auto delta_final_default = 2.0 * _GIIc / _S;

  if (out)
    _to = neml2::where(pos_mask, delta_final_open, Scalar(delta_final_default));

  if (dout_din)
  {
    // d(delta_final)/d(delta_init) = -delta_final / delta_init  (opening only)
    const auto ddf_dinit_open = -delta_final_open / _delta_init();

    // d(beta_sq_ratio)/d(beta) = 2 beta / (1+beta^2)^2
    const auto dbeta_sq_ratio_dbeta = 2.0 * _beta() / (one_plus_beta_sq * one_plus_beta_sq);
    // d(delta_final)/d(beta) = (2/(K delta_init)) (GIIc-GIc) eta (r+eps)^(eta-1) dr/dbeta
    const auto ddf_dbeta_open = (2.0 / (_K * _delta_init())) * (_GIIc - _GIc) * _eta *
                                neml2::pow(pow_base, _eta - 1.0) * dbeta_sq_ratio_dbeta;

    const auto zero = Scalar::zeros_like(_beta());
    _to.d(_delta_init) = neml2::where(pos_mask, ddf_dinit_open, zero);
    _to.d(_beta) = neml2::where(pos_mask, ddf_dbeta_open, zero);
    _to.d(_dn_pos) = Scalar::zeros_like(_dn_pos());
  }
}
} // namespace neml2
