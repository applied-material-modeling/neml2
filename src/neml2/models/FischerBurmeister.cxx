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

#include "neml2/models/FischerBurmeister.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(FischerBurmeister);
OptionSet
FischerBurmeister::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Fischer Burmeister Complementary Function: if \\f$ a \\ge 0, b \\ge 0, ab = 0 \\f$ then \\f$"
      "a+b-\\sqrt(a^2+b^2) = 0 \\f$";

  options.set_input("condition_a") = VariableName("state", "a");
  options.set("condition_a").doc() = "Condition a";

  options.set_input("condition_b") = VariableName("state", "b");
  options.set("condition_b").doc() = "Condition b";

  options.set_output("fischer_burmeister") = VariableName("state", "fb");
  options.set("fischer_burmeister").doc() = "Fischer Burmeister condition";

  return options;
}

FischerBurmeister::FischerBurmeister(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("condition_a")),
    _b(declare_input_variable<Scalar>("condition_b")),
    _fb(declare_output_variable<Scalar>("fischer_burmeister"))
{
}

void
FischerBurmeister::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _fb = _a + _b - sqrt(_a * _a + _b * _b);
  }

  if (dout_din)
  {
    _fb.d(_a) = 1.0 - _a / sqrt(_a * _a + _b * _b + machine_precision());
    _fb.d(_b) = 1.0 - _b / sqrt(_a * _a + _b * _b + machine_precision());
  }
}
}