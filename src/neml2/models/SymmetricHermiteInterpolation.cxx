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

#include "neml2/models/SymmetricHermiteInterpolation.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(SymmetricHermiteInterpolation);

OptionSet
SymmetricHermiteInterpolation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Define the symmetric Hermite interpolation function, taking the form of \\f$ (3c^2-4c^3) "
      "\\f$ for \\f$ 0 le c le 0.5 \\f$; \\f$ 3(1-c)^2 - 4(1-c)^3 \\f$ for \\f$ 0.5 le c le 1 "
      "\\f$, and 0.0 otherwise. Here, \\f$ c = \\frac{x-x_l}{x_h-x_l} \\f$ where \\f$x_l\\f$ and "
      "\\f$x_h\\f$ are the lower and upper bound for rescaling the input argument.";

  options.set_input("argument");
  options.set("argument").doc() = "Argument of the smooth step function";

  options.set<bool>("define_second_derivatives") = true;

  options.set_output("value");
  options.set("value").doc() = "Value of the smooth step function";

  options.set_buffer<TensorName<Scalar>>("lower_bound");
  options.set("lower_bound").doc() = "Lower bound of the argument";

  options.set_buffer<TensorName<Scalar>>("upper_bound");
  options.set("upper_bound").doc() = "Upper bound of the argument";

  return options;
}

SymmetricHermiteInterpolation::SymmetricHermiteInterpolation(const OptionSet & options)
  : Model(options),
    _x(declare_input_variable<Scalar>("argument")),
    _y(declare_output_variable<Scalar>("value")),
    _x0(declare_buffer<Scalar>("lb", "lower_bound")),
    _x1(declare_buffer<Scalar>("ub", "upper_bound"))
{
}

void
SymmetricHermiteInterpolation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto eps = machine_precision(_x.scalar_type()).toDouble();
  const auto x = clamp((_x - _x0) / (_x1 - _x0), eps, 1.0 - eps);

  if (out)
  {
    auto f_xl = 3 * x * x - 4 * x * x * x;
    auto f_xh = 3 * (1 - x) * (1 - x) - 4 * (1 - x) * (1 - x) * (1 - x);
    _y = where(x < 0.5, f_xl, f_xh);
  }

  if (dout_din)
  {
    auto df_xl = 6 * x - 12 * x * x;
    auto df_xh = -6 * (1 - x) + 12 * (1 - x) * (1 - x);

    _y.d(_x) = where(x < 0.5, df_xl, df_xh);
  }

  if (d2out_din2)
  {
    auto df2_xl = 6 - 24 * x;
    auto df2_xh = 6 - 24 * (1 - x);

    const auto zeromask = Scalar(at::logical_and(at::lt(x, 1.0 - eps), at::gt(x, eps)));
    _y.d(_x, _x) = zeromask * where(x < 0.5, df2_xl, df2_xh);
  }
}
} // namespace neml2
