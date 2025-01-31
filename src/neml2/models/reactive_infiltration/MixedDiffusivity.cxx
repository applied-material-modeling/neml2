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

#include "neml2/models/reactive_infiltration/MixedDiffusivity.h"

namespace neml2
{
register_NEML2_object(MixedDiffusivity);

OptionSet
MixedDiffusivity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the liquid phase volume fraction";

  options.set_input("liquid_fraction") = VariableName{"state", "phi_l"};
  options.set("liquid_fraction").doc() = "Volume fraction of the liquid phase";
  options.set_input("product_fraction") = VariableName{"state", "phi_p"};
  options.set("product_fraction").doc() = "Volume fraction of the product phase";

  options.set_output("diffusivity") = VariableName{"state", "D"};
  options.set("diffusivity").doc() = "Diffusivity";

  options.set_parameter<CrossRef<Scalar>>("liquid_diffusivity");
  options.set("liquid_diffusivity").doc() = "Diffusivity of the liquid species in the liquid phase";
  options.set_parameter<CrossRef<Scalar>>("product_diffusivity");
  options.set("product_diffusivity").doc() =
      "Diffusivity of the liquid species in the product phase";

  return options;
}

MixedDiffusivity::MixedDiffusivity(const OptionSet & options)
  : Model(options),
    _phi_l(declare_input_variable<Scalar>("liquid_fraction")),
    _phi_p(declare_input_variable<Scalar>("product_fraction")),
    _D(declare_output_variable<Scalar>("diffusivity")),
    _D_l(declare_parameter<Scalar>("D_l", "liquid_diffusivity")),
    _D_p(declare_parameter<Scalar>("D_p", "product_diffusivity"))
{
}

void
MixedDiffusivity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivatives not implemented");

  const auto f = _phi_p / (_phi_l + _phi_p + machine_precision());

  if (out)
  {
    _D = _D_l + (_D_p - _D_l) * f;
  }

  if (dout_din)
  {
    const auto denom2 =
        (_phi_l + _phi_p + machine_precision()) * (_phi_l + _phi_p + machine_precision());
    _D.d(_phi_l) = -(_D_p - _D_l) * _phi_p / denom2;
    _D.d(_phi_p) = (_D_p - _D_l) * _phi_l / denom2;
  }
}
} // namespace neml2
