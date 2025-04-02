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

#include "neml2/models/R2Multiplication.h"
#include "neml2/tensors/R4.h"

namespace neml2
{
register_NEML2_object(R2Multiplication);

OptionSet
R2Multiplication::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Multiplication of form \\f$ A B \\f$, where \\f$ A \ff$ and \\f$ B \ff$ are "
                  "second order tensors.";

  options.set<VariableName>("A");
  options.set("A").doc() = "Variable \\f$ A \\f$";

  options.set<bool>("invert_A") = false;
  options.set("invert_A").doc() = "Whether to invert \\f$ A \\f$";

  options.set<VariableName>("B");
  options.set("B").doc() = "Variable \\f$ B \\f$";

  options.set<bool>("invert_B") = false;
  options.set("invert_B").doc() = "Whether to invert \\f$ B \\f$";

  options.set_output("to");
  options.set("to").doc() = "The result of the multiplication";

  return options;
}

R2Multiplication::R2Multiplication(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<R2>("to")),
    _A(declare_input_variable<R2>("A")),
    _B(declare_input_variable<R2>("B")),
    _invA(options.get<bool>("invert_A")),
    _invB(options.get<bool>("invert_B"))
{
}

void
R2Multiplication::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto A = _invA ? R2(_A).inverse() : _A;
  const auto B = _invB ? R2(_B).inverse() : _B;

  if (out)
    _to = A * B;

  if (dout_din)
  {
    const auto I = R2::identity(_A.options());

    if (_invA)
      _to.d(_A) = -R4(at::einsum("...ip,...qj", {A, A * B}));
    else
      _to.d(_A) = R4(at::einsum("...im,...nj", {I, B}));

    if (_invB)
      _to.d(_B) = -R4(at::einsum("...ip,...qj", {A * B, B}));
    else
      _to.d(_B) = R4(at::einsum("...im,...nj", {A, I}));
  }
}
} // namespace neml2
