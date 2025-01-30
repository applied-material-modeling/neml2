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

#include "neml2/models/reactive_infiltration/LiquidFraction.h"

namespace neml2
{
register_NEML2_object(LiquidFraction);

OptionSet
LiquidFraction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the liquid phase volume fraction";

  options.set_input("liquid_concentration") = VariableName{"forces", "alpha"};
  options.set("liquid_concentration").doc() = "Total concentration of the liquid species";
  options.set_input("product_fraction") = VariableName{"state", "phi_p"};
  options.set("product_fraction").doc() = "Volume fraction of the product phase";

  options.set_output("liquid_fraction") = VariableName{"state", "phi_l"};
  options.set("liquid_fraction").doc() = "Volume fraction of the liquid phase";

  options.set<Real>("liquid_molar_volume");
  options.set("liquid_molar_volume").doc() = "Molar volume of the species in the liquid phase";
  options.set<Real>("solid_molar_volume");
  options.set("solid_molar_volume").doc() = "Molar volume of the species in the solid phase";

  return options;
}

LiquidFraction::LiquidFraction(const OptionSet & options)
  : Model(options),
    _alpha(declare_input_variable<Scalar>("liquid_concentration")),
    _phi_p(declare_input_variable<Scalar>("product_fraction")),
    _phi_l(declare_output_variable<Scalar>("liquid_fraction")),
    _omega_l(options.get<Real>("liquid_molar_volume")),
    _omega_s(options.get<Real>("solid_molar_volume"))
{
}

void
LiquidFraction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivatives not implemented");

  if (out)
  {
    _phi_l = _alpha * _omega_l - _phi_p * _omega_l / (_omega_l + _omega_s);
  }

  if (dout_din)
  {
    _phi_l.d(_alpha) = Scalar::full(_omega_l, _phi_l.options());
    _phi_l.d(_phi_p) = Scalar::full(-_omega_l / (_omega_l + _omega_s), _phi_l.options());
  }
}
} // namespace neml2
