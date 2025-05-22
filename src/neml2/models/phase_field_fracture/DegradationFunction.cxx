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

#include "neml2/models/phase_field_fracture/DegradationFunction.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
OptionSet
DegradationFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Base class for degradation function to degrade the elastic strain energy density";

  options.set_input("damage") = VariableName(STATE, "d");
  options.set("damage").doc() = "Damage/Phase-field variable";

  options.set_output("degradation") = VariableName(STATE, "g");
  options.set("degradation").doc() = "Value of the dedgradation function";

  return options;
}

DegradationFunction::DegradationFunction(const OptionSet & options)
  : Model(options),
    _d(declare_input_variable<Scalar>("damage")),
    _g(declare_output_variable<Scalar>("degradation"))
{
}
} // namespace neml2
