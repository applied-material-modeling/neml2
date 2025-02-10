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

#include "neml2/models/solid_mechanics/GursonCavitation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(GursonCavitation);

OptionSet
GursonCavitation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Local mass balance used in conjunction with the GTNYieldFunction, \\f$ "
                  "\\dot{\\phi} = (1-\\phi) \\dot{\\varepsilon}_p \\f$.";

  options.set_input("plastic_strain_rate") = VariableName(STATE, "internal", "Ep_rate");
  options.set("plastic_strain_rate").doc() = "Plastic strain rate";

  options.set_input("void_fraction") = VariableName(STATE, "internal", "f");
  options.set("void_fraction").doc() = "Void fraction (porosity)";

  options.set_output("void_fraction_rate") = VariableName(STATE, "internal", "f_rate");
  options.set("void_fraction_rate").doc() = "Rate of void evolution";

  return options;
}

GursonCavitation::GursonCavitation(const OptionSet & options)
  : Model(options),
    _phi_dot(declare_output_variable<Scalar>("void_fraction_rate")),
    _Ep_dot(declare_input_variable<SR2>("plastic_strain_rate")),
    _phi(declare_input_variable<Scalar>("void_fraction"))
{
}

void
GursonCavitation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto ep_dot = SR2(_Ep_dot).tr();

  if (out)
    _phi_dot = (1 - _phi) * ep_dot;

  if (dout_din)
  {
    const auto I = SR2::identity(_phi.options());

    if (_phi.is_dependent())
      _phi_dot.d(_phi) = -ep_dot;

    if (_Ep_dot.is_dependent())
      _phi_dot.d(_Ep_dot) = I * (1 - _phi);
  }
}
} // namespace neml2
