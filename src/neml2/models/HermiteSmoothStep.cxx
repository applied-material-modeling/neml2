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

#include "neml2/models/HermiteSmoothStep.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(HermiteSmoothStep);

OptionSet
HermiteSmoothStep::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "The smooth step function defined by Hermite polynomials";

  options.set_input("argument");
  options.set("argument").doc() = "Argument of the smooth step function";

  options.set_output("value");
  options.set("value").doc() = "Value of the smooth step function";

  options.set_buffer<TensorName<Scalar>>("lower_bound");
  options.set("lower_bound").doc() = "Lower bound of the argument";

  options.set_buffer<TensorName<Scalar>>("upper_bound");
  options.set("upper_bound").doc() = "Upper bound of the argument";

  options.set<bool>("complement_condition") = false;
  options.set("complement_condition").doc() = "Whether takes 1 to subtract the function.";

  return options;
}

HermiteSmoothStep::HermiteSmoothStep(const OptionSet & options)
  : Model(options),
    _x(declare_input_variable<Scalar>("argument")),
    _y(declare_output_variable<Scalar>("value")),
    _x0(declare_buffer<Scalar>("lb", "lower_bound")),
    _x1(declare_buffer<Scalar>("ub", "upper_bound")),
    _comp_cond(options.get<bool>("complement_condition"))
{
}

void
HermiteSmoothStep::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivatives not implemented");

  const auto eps = machine_precision(_x.scalar_type()).toDouble();
  const auto x = clamp((_x - _x0) / (_x1 - _x0), eps, 1.0 - eps);

  if (out)
  {

    _y = 3 * x * x - 2 * x * x * x;
    if (_comp_cond)
      _y = 1 - (3 * x * x - 2 * x * x * x);
  }

  if (dout_din)
  {
    _y.d(_x) = 6 * x * (1 - x) / (_x1 - _x0);
    if (_comp_cond)
      _y.d(_x) = -(6 * x * (1 - x) / (_x1 - _x0));
  }
}
} // namespace neml2
