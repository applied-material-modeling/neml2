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

#include "neml2/models/pyrolysis/PyrolysisConversionAmount.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(PyrolysisConversionAmount);
OptionSet
PyrolysisConversionAmount::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the conversion amount from the pyrolysis process, defined by the ratio between "
      "the mass loss and the total possible mass loss from the pyrolysis.";

  options.set_parameter<TensorName<Scalar>>("initial_solid_mass_fraction");
  options.set("initial_solid_mass_fraction").doc() =
      "The solid's initial mass fraction before the pyrolysis";

  options.set_parameter<TensorName<Scalar>>("initial_binder_mass_fraction");
  options.set("initial_binder_mass_fraction").doc() =
      "The binder's initial mass fraction before the pyrolysis";

  options.set_parameter<TensorName<Scalar>>("reaction_yield");
  options.set("reaction_yield").doc() =
      "The final reaction yield from the pyrolysis process (between 0 and 1)";

  options.set_input("solid_mass_fraction") = VariableName("state", "solid_mass_fraction");
  options.set("solid_mass_fraction").doc() = "The solid's mass fraction.";

  options.set_output("reaction_amount") = VariableName("state", "reaction_amount");
  options.set("reaction_amount").doc() = "The amount of converted mass from the pyrolysis.";

  return options;
}

PyrolysisConversionAmount::PyrolysisConversionAmount(const OptionSet & options)
  : Model(options),
    _ws0(declare_parameter<Scalar>("ws0", "initial_solid_mass_fraction")),
    _wb0(declare_parameter<Scalar>("wb0", "initial_binder_mass_fraction")),
    _Y(declare_parameter<Scalar>("Y", "reaction_yield")),
    _ws(declare_input_variable<Scalar>("solid_mass_fraction")),
    _a(declare_output_variable<Scalar>("reaction_amount"))
{
}

void
PyrolysisConversionAmount::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto mass_ini = 1 + _ws0 / _wb0;

  if (out)
  {
    _a = (mass_ini - _ws / _wb0) / (1.0 - _Y);
  }

  if (dout_din)
  {
    _a.d(_ws) = -1.0 / (_wb0 * (1.0 - _Y));
  }
}
}