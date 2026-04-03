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

#include "neml2/models/common/FischerBurmeister.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(FischerBurmeister);
OptionSet
FischerBurmeister::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "If \\f$ a \\ge 0, b \\ge 0, ab = 0 \\f$ then the Fischer Burmeister (FB) "
                  "complementarity condition is:\\f$ r = a+b-\\sqrt(a^2+b^2) = 0 \\f$.";

  options.add_input("a", "First condition");
  options.add_input("b", "Second condition");
  options.add_output("complementarity", "The Fischer Burmeister complementarity condition");

  EnumSelection ineq({"GE", "LE"}, "GE");
  options.add<EnumSelection>(
      "a_inequality", ineq, "Type of the inequality for the first variable a.");
  options.add<EnumSelection>("b_inequality", ineq, "Type of inequality for the second variable b.");

  return options;
}

FischerBurmeister::FischerBurmeister(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("a")),
    _b(declare_input_variable<Scalar>("b")),
    _a_ineq(options.get<EnumSelection>("a_inequality")),
    _b_ineq(options.get<EnumSelection>("b_inequality")),
    _c(declare_output_variable<Scalar>("complementarity"))
{
}

void
FischerBurmeister::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto ia = 1.0;
  if (_a_ineq == "LE")
    ia = -1.0;

  auto ib = 1.0;
  if (_b_ineq == "LE")
    ib = -1.0;

  if (out)
    _c = _a * ia + _b * ib - sqrt(_a * _a + _b * _b);

  if (dout_din)
  {
    const auto eps = machine_precision(_a.scalar_type());
    _c.d(_a) = ia - _a / sqrt(_a * _a + _b * _b + eps);
    _c.d(_b) = ib - _b / sqrt(_a * _a + _b * _b + eps);
  }
}
}
