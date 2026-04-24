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

#include "neml2/models/common/HermiteSmoothStep.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/functions/clamp.h"

namespace neml2
{
register_NEML2_object(HermiteSmoothStep);

OptionSet
HermiteSmoothStep::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "The smooth step function defined by Hermite polynomials";

  options.add_input("argument", "Argument of the smooth step function");
  options.add_output("value", "Value of the smooth step function");
  options.add_buffer<Scalar>("lower_bound", "Lower bound of the argument");
  options.add_buffer<Scalar>("upper_bound", "Upper bound of the argument");

  options.add<bool>(
      "complement",
      false,
      "Whether to take the complement of the smooth step function, i.e. 1 - smooth_step");

  return options;
}

HermiteSmoothStep::HermiteSmoothStep(const OptionSet & options)
  : Model(options),
    _x(declare_input_variable<Scalar>("argument")),
    _y(declare_output_variable<Scalar>("value")),
    _x0(declare_buffer<Scalar>("lb", "lower_bound")),
    _x1(declare_buffer<Scalar>("ub", "upper_bound")),
    _comp_cond(options.get<bool>("complement"))
{
}

void
HermiteSmoothStep::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_x.scalar_type());
  const auto x = clamp((_x - _x0) / (_x1 - _x0), eps, 1.0 - eps);

  if (out)
  {
    auto y = 3 * x * x - 2 * x * x * x;
    _y = _comp_cond ? 1 - y : y;
  }

  if (dout_din)
  {
    auto dy_dx = 6 * x * (1 - x) / (_x1 - _x0);
    _y.d(_x) = _comp_cond ? -dy_dx : dy_dx;
  }
}
} // namespace neml2
