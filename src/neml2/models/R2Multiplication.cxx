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
#include "neml2/tensors/functions/inv.h"

namespace neml2
{
register_NEML2_object(R2Multiplication);

OptionSet
R2Multiplication::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Multiplication of form \\f$ A B \\f$, where \\f$ A \\f$ and \\f$ B \\f$ are "
                  "second order tensors. A and B can be inverted and/or transposed per request.";

  options.set<VariableName>("A");
  options.set("A").doc() = "Variable A";

  options.set<bool>("invert_A") = false;
  options.set("invert_A").doc() = "Whether to invert A";

  options.set<bool>("transpose_A") = false;
  options.set("transpose_A").doc() = "Whether to transpose A";

  options.set<VariableName>("B");
  options.set("B").doc() = "Variable B";

  options.set<bool>("invert_B") = false;
  options.set("invert_B").doc() = "Whether to invert B";

  options.set<bool>("transpose_B") = false;
  options.set("transpose_B").doc() = "Whether to transpose B";

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
    _invB(options.get<bool>("invert_B")),
    _transA(options.get<bool>("transpose_A")),
    _transB(options.get<bool>("transpose_B"))
{
}

void
R2Multiplication::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto A = _invA ? neml2::inv(_A()) : _A();
  if (_transA)
    A = A.transpose();

  auto B = _invB ? neml2::inv(_B()) : _B();
  if (_transB)
    B = B.transpose();

  const auto AB = A * B;

  if (out)
    _to = AB;

  if (dout_din)
  {
    const auto I = R2::identity(_A.options());

    if (_invA)
    {
      if (_transA)
        _to.d(_A) = -A.base_unsqueeze(-2).base_unsqueeze(-3) *
                    AB.transpose().base_unsqueeze(-3).base_unsqueeze(-1);
      else
        _to.d(_A) = -A.base_unsqueeze(-2).base_unsqueeze(-1) *
                    AB.transpose().base_unsqueeze(-3).base_unsqueeze(-2);
    }
    else
    {
      if (_transA)
        _to.d(_A) = I.base_unsqueeze(-2).base_unsqueeze(-3) *
                    B.transpose().base_unsqueeze(-3).base_unsqueeze(-1);
      else
        _to.d(_A) = I.base_unsqueeze(-2).base_unsqueeze(-1) *
                    B.transpose().base_unsqueeze(-3).base_unsqueeze(-2);
    }

    if (_invB)
    {
      if (_transB)
        _to.d(_B) = -AB.base_unsqueeze(-2).base_unsqueeze(-3) *
                    B.transpose().base_unsqueeze(-3).base_unsqueeze(-1);
      else
        _to.d(_B) = -AB.base_unsqueeze(-2).base_unsqueeze(-1) *
                    B.transpose().base_unsqueeze(-3).base_unsqueeze(-2);
    }
    else
    {
      if (_transB)
        _to.d(_B) = A.base_unsqueeze(-2).base_unsqueeze(-3) *
                    I.transpose().base_unsqueeze(-3).base_unsqueeze(-1);
      else
        _to.d(_B) = A.base_unsqueeze(-2).base_unsqueeze(-1) *
                    I.transpose().base_unsqueeze(-3).base_unsqueeze(-2);
    }
  }
}
} // namespace neml2
