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

#include "neml2/models/common/IrreversibleScalar.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(IrreversibleScalar);

OptionSet
IrreversibleScalar::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Monotonically-increasing ratchet on a Scalar: \\f$ y = \\max(y_{n-1}, x) \\f$ "
                  "where \\f$ x \\f$ is the trial value (`from`) and \\f$ y_{n-1} \\f$ is the "
                  "previous-step value of the output (auto-declared via `history_name`).";

  options.add_input("from", "Trial value at the current step");
  options.add_output("to", "Updated (irreversibly-capped) value");

  return options;
}

IrreversibleScalar::IrreversibleScalar(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Scalar>("to")),
    _from(declare_input_variable<Scalar>("from")),
    _to_old(declare_input_variable<Scalar>(history_name(_to.name(), /*nstep=*/1)))
{
}

void
IrreversibleScalar::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Detach the mask so (1) `where` doesn't try to backprop through its condition and
  // (2) the TorchScript tracer doesn't reject a grad-tracking mask captured into the JIT graph.
  const auto advance_mask = (_from() > _to_old()).detach();

  if (out)
    _to = neml2::where(advance_mask, _from(), _to_old());

  if (dout_din)
  {
    const auto one = Scalar::ones_like(_from());
    const auto zero = Scalar::zeros_like(_from());
    _to.d(_from) = neml2::where(advance_mask, one, zero);
    _to.d(_to_old) = neml2::where(advance_mask, zero, one);
  }
}
} // namespace neml2
