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

#include "neml2/models/pyrolysis/PyrolysisConversionAmountStateful.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(PyrolysisConversionAmountStateful);
OptionSet
PyrolysisConversionAmountStateful::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the conversion amount from the pyrolysis process, defined by the ratio between "
      "the mass loss and the total possible mass loss from the pyrolysis.";

  options.set_input("initial_mass_solid") = VariableName("state", "initial_mass_solid");
  options.set("initial_mass_solid").doc() = "The solid's initial mass before the pyrolysis";

  options.set_input("initial_mass_binder") = VariableName("state", "initial_mass_binder");
  options.set("initial_mass_binder").doc() = "The binder's initial mas before the pyrolysis";

  options.set_parameter<TensorName<Scalar>>("reaction_yield");
  options.set("reaction_yield").doc() =
      "The final reaction yield from the pyrolysis process (between 0 and 1)";

  options.set_input("mass_solid") = VariableName("state", "mass_solid");
  options.set("mass_solid").doc() = "The solid's mass.";

  options.set_output("reaction_amount") = VariableName("state", "reaction_amount");
  options.set("reaction_amount").doc() = "The amount of converted mass from the pyrolysis.";

  return options;
}

PyrolysisConversionAmountStateful::PyrolysisConversionAmountStateful(const OptionSet & options)
  : Model(options),
    _ms0(declare_input_variable<Scalar>("initial_mass_solid")),
    _mb0(declare_input_variable<Scalar>("initial_mass_binder")),
    _Y(declare_parameter<Scalar>("Y", "reaction_yield")),
    _ms(declare_input_variable<Scalar>("mass_solid")),
    _a(declare_output_variable<Scalar>("reaction_amount"))
{
}

void
PyrolysisConversionAmountStateful::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto mass_ini = 1 + _ms0 / _mb0;

  if (out)
  {
    _a = (mass_ini - _ms / _mb0) / (1.0 - _Y);
  }

  if (dout_din)
  {
    _a.d(_ms) = -1.0 / (_mb0 * (1.0 - _Y));
    _a.d(_ms0) = 1.0 / (_mb0 * (1.0 - _Y));
    _a.d(_mb0) = -(_ms0 - _ms) / (_mb0 * _mb0 * (1.0 - _Y));
  }
}
}