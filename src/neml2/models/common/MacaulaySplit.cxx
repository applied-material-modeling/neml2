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

#include "neml2/models/common/MacaulaySplit.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/macaulay.h"

namespace neml2
{
register_NEML2_object(MacaulaySplit);

OptionSet
MacaulaySplit::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Split a Scalar into its Macaulay (positive) and negative parts: "
                  "\\f$ \\langle x \\rangle_+ = \\max(x, 0) \\f$ and "
                  "\\f$ \\langle x \\rangle_- = x - \\langle x \\rangle_+ \\f$.";

  options.add_input("from", "The Scalar to split");
  options.add_output("to_positive", "Name of the Macaulay (positive) part output");
  options.add_output("to_negative", "Name of the negative-part output");

  return options;
}

MacaulaySplit::MacaulaySplit(const OptionSet & options)
  : Model(options),
    _from(declare_input_variable<Scalar>("from")),
    _to_pos(declare_output_variable<Scalar>("to_positive")),
    _to_neg(declare_output_variable<Scalar>("to_negative"))
{
}

void
MacaulaySplit::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto x = _from();
  const auto x_pos = neml2::macaulay(x);

  if (out)
  {
    _to_pos = x_pos;
    _to_neg = x - x_pos;
  }

  if (dout_din)
  {
    const auto H = neml2::heaviside(x);
    _to_pos.d(_from) = H;
    _to_neg.d(_from) = 1.0 - H;
  }
}
} // namespace neml2
