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

#include "neml2/models/ScalartoDiagSR2.h"
#include "neml2/tensors/SR2.h"

namespace neml2
{
register_NEML2_object(ScalartoDiagSR2);

OptionSet
ScalartoDiagSR2::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Project the scalar to the components of a Diagonal Symmetric Rank 2 tensor";

  options.set<bool>("define_second_derivatives") = true;

  options.set_input("input");
  options.set("input").doc() = "Symmetric tensor to convert";

  options.set_output("output");
  options.set("output").doc() = "Output full rank two tensor";

  return options;
}

ScalartoDiagSR2::ScalartoDiagSR2(const OptionSet & options)
  : Model(options),
    _input(declare_input_variable<Scalar>("input")),
    _output(declare_output_variable<SR2>("output"))
{
}

void
ScalartoDiagSR2::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto I = SR2::identity(_input.options());

  if (out)
    _output = _input * I;

  if (dout_din)
    _output.d(_input) = I;

  // Second derivative is zero
  (void)d2out_din2;
}
} // namespace neml2
