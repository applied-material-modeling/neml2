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

#include "neml2/models/solid_mechanics/traction_separation_law/CamanhoDavilaInitiation.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(CamanhoDavilaInitiation);

OptionSet
CamanhoDavilaInitiation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Camanho-Davila mixed-mode initiation displacement jump. Opening branch: "
      "\\f$ \\delta_\\text{init} = \\delta_{n0} \\delta_{s0} \\sqrt{1+\\beta^2} / "
      "\\sqrt{\\delta_{s0}^2 + \\beta^2 \\delta_{n0}^2} \\f$ with \\f$ \\delta_{n0} = N/K \\f$ "
      "and \\f$ \\delta_{s0} = S/K \\f$. Compression branch: \\f$ \\delta_\\text{init} = "
      "\\delta_{s0} \\f$.";

  options.add_input("mixity", "Mode-mixity ratio \\f$ \\beta \\f$");
  options.add_input("normal",
                    "Macaulay-positive normal jump \\f$ \\delta_n^+ \\f$ (used only to "
                    "determine the opening / compression branch)");
  options.add_output("to", "Initiation displacement jump \\f$ \\delta_\\text{init} \\f$");
  options.add_parameter<Scalar>("penalty_stiffness", "Penalty stiffness K");
  options.add_parameter<Scalar>("normal_strength", "Tensile (normal) strength N");
  options.add_parameter<Scalar>("shear_strength", "Shear strength S");

  return options;
}

CamanhoDavilaInitiation::CamanhoDavilaInitiation(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to")),
    _beta(declare_input_variable<Scalar>("mixity")),
    _dn_pos(declare_input_variable<Scalar>("normal")),
    // allow_nonlinear=false: this model does not provide _to.d(*K) / etc.
    _K(declare_parameter<Scalar>("K", "penalty_stiffness", false)),
    _N(declare_parameter<Scalar>("N", "normal_strength", false)),
    _S(declare_parameter<Scalar>("S", "shear_strength", false))
{
}

void
CamanhoDavilaInitiation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_beta.scalar_type());
  const auto pos_mask = (_dn_pos() > 0.0).detach();

  const auto delta_n0 = _N / _K;
  const auto delta_s0 = _S / _K;

  const auto beta_sq = _beta() * _beta();
  const auto delta_mixed_init_sq = delta_s0 * delta_s0 + beta_sq * delta_n0 * delta_n0 + eps;
  const auto delta_mixed_init = neml2::sqrt(delta_mixed_init_sq);
  const auto delta_init_open = delta_n0 * delta_s0 * neml2::sqrt(1.0 + beta_sq) / delta_mixed_init;

  if (out)
    _to = neml2::where(pos_mask, delta_init_open, Scalar(delta_s0));

  if (dout_din)
  {
    // d(delta_init)/d(beta) = delta_init * beta * [1/(1+beta^2) - delta_n0^2/delta_mixed^2]
    const auto ddelta_init_dbeta_open =
        delta_init_open * _beta() *
        (1.0 / (1.0 + beta_sq) - delta_n0 * delta_n0 / delta_mixed_init_sq);
    _to.d(_beta) = neml2::where(pos_mask, ddelta_init_dbeta_open, Scalar::zeros_like(_beta()));
    // delta_init does not depend on dn_pos within either branch — only the branch selection does,
    // and that's a Heaviside whose derivative is a Dirac (zero almost everywhere).
    _to.d(_dn_pos) = Scalar::zeros_like(_dn_pos());
  }
}
} // namespace neml2
