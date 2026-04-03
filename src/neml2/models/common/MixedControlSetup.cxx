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

#include "neml2/models/common/MixedControlSetup.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/diagonalize.h"

namespace neml2
{
register_NEML2_object(MixedControlSetup);

OptionSet
MixedControlSetup::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "'Mix' two variables \\f$ \\underline{x} \\f$ and \\f$ \\bar{x} \\f$ into a single output "
      "variable \\f$ y \\f$.  For each component of the control signal, if the control is above "
      "the threshold, then the corresponding component of \\f$ y \\f$ is set to the corresponding "
      "component of \\f$ \\bar{x} \\f$, otherwise it is set to the corresponding component of \\f$ "
      "\\underline{x} \\f$.  Values not selected by the control signal are output to \\f$ z \\f$.";

  options.add_input("control", "The control signal.");
  options.add_buffer<SR2>(
      "threshold", TensorName<SR2>("0.5"), "The threshold to switch between the two control");

  options.add_input(
      "x_above",
      "The variable whose values are selected when the control signal is greater than the "
      "threshold.");
  options.add_input(
      "x_below",
      "The variable whose values are selected when the control signal is less than or equal to the "
      "threshold.");

  options.add_output(
      "y", "The output variable holding the selected values based on the control signal.");
  options.add_output(
      "z", "The output variable holding the non-selected values based on the control signal.");

  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _threshold(declare_buffer<SR2>("c", "threshold")),
    _control(declare_input_variable<SR2>("control")),
    _x_above(declare_input_variable<SR2>("x_above")),
    _x_below(declare_input_variable<SR2>("x_below")),
    _y(declare_output_variable<SR2>("y")),
    _z(declare_output_variable<SR2>("z"))
{
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto [dbelow, dabove] = make_operators(_control());

  if (out)
  {
    _y = dabove * _x_above + dbelow * _x_below;
    _z = dbelow * _x_above + dabove * _x_below;
  }

  if (dout_din)
  {
    _y.d(_x_above) = dabove;
    _z.d(_x_above) = dbelow;
    _y.d(_x_below) = dbelow;
    _z.d(_x_below) = dabove;
  }
}

std::pair<SSR4, SSR4>
MixedControlSetup::make_operators(const SR2 & control) const
{
  auto below = control <= _threshold;
  auto above = control > _threshold;

  // convert to matching dtype
  auto ones_below = below.to(control.options());
  auto ones_above = 1.0 - ones_below;

  auto dbelow = base_diagonalize(ones_below);
  auto dabove = base_diagonalize(ones_above);

  return {dbelow, dabove};
}

} // namespace neml2
