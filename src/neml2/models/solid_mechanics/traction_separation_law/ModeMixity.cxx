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

#include "neml2/models/solid_mechanics/traction_separation_law/ModeMixity.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(ModeMixity);

OptionSet
ModeMixity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Mode-mixity ratio \\f$ \\beta = \\delta_s / \\delta_n^+ \\f$ in the opening "
                  "branch; \\f$ \\beta = 0 \\f$ in compression. Uses a safe-divisor "
                  "`where`-and-detach pattern so the masked-off compression branch does not "
                  "trip on a division by zero.";

  options.add_input("normal_separation",
                    "Normal separation (typically the Macaulay-positive part of the normal jump)");
  options.add_input("tangential_separation", "Tangential separation magnitude");
  options.add_output("mode_mixity", "Mode-mixity ratio");

  return options;
}

ModeMixity::ModeMixity(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("mode_mixity")),
    _dn(declare_input_variable<Scalar>("normal_separation")),
    _ds(declare_input_variable<Scalar>("tangential_separation"))
{
}

void
ModeMixity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto pos_mask = (_dn() > 0.0).detach();
  const auto one = Scalar::ones_like(_dn());
  const auto zero = Scalar::zeros_like(_dn());
  // Safe denominator: 1 in the masked-off branch so beta_open division doesn't trip on 0.
  const auto safe_dn = neml2::where(pos_mask, _dn(), one);
  const auto inv_dn = 1.0 / safe_dn;

  if (out)
    _to = neml2::where(pos_mask, _ds() * inv_dn, zero);

  if (dout_din)
  {
    // d(beta)/d(delta_s)     = 1 / delta_n  (opening) ; 0 (compression)
    // d(beta)/d(delta_n_pos) = -delta_s / delta_n^2  (opening) ; 0 (compression)
    _to.d(_ds) = neml2::where(pos_mask, inv_dn, zero);
    _to.d(_dn) = neml2::where(pos_mask, -_ds() * inv_dn * inv_dn, zero);
  }
}
} // namespace neml2
