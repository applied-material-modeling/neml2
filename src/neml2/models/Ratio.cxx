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

#include "neml2/models/Ratio.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(Ratio);
OptionSet
Ratio::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the ratio between two variables";

  options.set_input("numerator") = VariableName("state", "numerator");
  options.set("numerator").doc() = "Numerator variable";

  options.set_input("denominator") = VariableName("state", "denominator");
  options.set("denominator").doc() = "Denominator variable";

  options.set_output("out") = VariableName("state", "out");
  options.set("out").doc() = "Ratio out.";

  return options;
}

Ratio::Ratio(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("numerator")),
    _b(declare_input_variable<Scalar>("denominator")),
    _aob(declare_output_variable<Scalar>("out"))
{
}

void
Ratio::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _aob = _a / _b;
  }

  if (dout_din)
  {
    _aob.d(_a) = 1 / _b;
    _aob.d(_b) = -1 * _a / (_b * _b);
  }
}
}