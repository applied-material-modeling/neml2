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

#include "neml2/models/Product.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(Product);
OptionSet
Product::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Define the product between two variables";

  options.set_input("variable_a") = VariableName("state", "variable_a");
  options.set("variable_a").doc() = "variable a";

  options.set_input("variable_b") = VariableName("state", "variable_b");
  options.set("variable_b").doc() = "variable b";

  options.set_output("out") = VariableName("state", "out");
  options.set("out").doc() = "Product out.";

  return options;
}

Product::Product(const OptionSet & options)
  : Model(options),
    _a(declare_input_variable<Scalar>("variable_a")),
    _b(declare_input_variable<Scalar>("variable_b")),
    _ab(declare_output_variable<Scalar>("out"))
{
}

void
Product::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _ab = _a * _b;
  }

  if (dout_din)
  {
    _ab.d(_a) = _b;
    _ab.d(_b) = _a;
  }
}
}
