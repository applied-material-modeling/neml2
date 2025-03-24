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
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(FischerBurmeister);
OptionSet
FischerBurmeister::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "By default,  if \\f$ a \\ge 0, "
                  "b \\ge 0, ab = 0 \\f$ then  the Fischer Burmeister (FB) condition is:\\f$"
                  "a+b-\\sqrt(a^2+b^2)\\f$, where a, b is the first_var and second_var "
                  "respectively and first_inequality = second_inequality = 'GE'. One could set "
                  "first_inequality = 'LE' (i.e. \\f$ a \\le 0, "
                  "b \\ge 0, ab = 0 \\f$, FB conditions is \\f$"
                  "-a+b-\\sqrt(a^2+b^2) \\f$). Same goes for second_inequality = 'LE'.";

  options.set_input("first_var") = VariableName(STATE, "a");
  options.set("first_var").doc() = "First condition";

  options.set_input("second_var") = VariableName(STATE, "b");
  options.set("second_var").doc() = "Second condition";

  EnumSelection conda({"GE", "LE"}, "GE");
  options.set<EnumSelection>("first_inequality") = conda;
  options.set("first_inequality").doc() = "Type of inequality for the first variable."
                                          "Default: GE. Options are " +
                                          conda.candidates_str();

  EnumSelection condb({"GE", "LE"}, "GE");
  options.set<EnumSelection>("second_inequality") = condb;
  options.set("second_inequality").doc() = "Type of inequality for the second variable."
                                           "Default: GE. Options are " +
                                           condb.candidates_str();

  options.set_output("fischer_burmeister") = VariableName(STATE, "fb");
  options.set("fischer_burmeister").doc() = "Fischer Burmeister condition";

  return options;
}

FischerBurmeister::FischerBurmeister(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("first_var")),
    _b(declare_input_variable<Scalar>("second_var")),
    _conda(options.get<EnumSelection>("first_inequality")),
    _condb(options.get<EnumSelection>("second_inequality")),
    _fb(declare_output_variable<Scalar>("fischer_burmeister"))
{
}

void
FischerBurmeister::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto ia = 1.0;
  if (_conda == "LE")
    ia = -1.0;

  auto ib = 1.0;
  if (_condb == "LE")
    ib = -1.0;

  if (out)
  {
    _fb = _a * ia + _b * ib - sqrt(_a * _a + _b * _b);
  }

  if (dout_din)
  {
    _fb.d(_a) = ia - _a / sqrt(_a * _a + _b * _b + machine_precision());
    _fb.d(_b) = ib - _b / sqrt(_a * _a + _b * _b + machine_precision());
  }
}
}