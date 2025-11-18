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

#include "neml2/models/MixedControlSetup.h"
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
      "Object to setup a model for mixed control.  Copies the values of the `fixed_values` and the "
      "`mixed_state` (the conjugate) into the input variables used by the model.";

  options.set_input("control") = VariableName(FORCES, "control");
  options.set("control").doc() = "The control signal.";

  options.set<TensorName<SR2>>("threshold") = TensorName<SR2>("0.5");
  options.set("threshold").doc() = "The threshold to switch between the two control";

  options.set_input("mixed_state") = VariableName(STATE, "mixed_state");
  options.set("mixed_state").doc() = "The name of the mixed state tensor. This holds the conjugate "
                                     "values to those being controlled";

  options.set_input("fixed_values") = VariableName(FORCES, "fixed_values");
  options.set("fixed_values").doc() =
      "The name of the fixed values, i.e. the actual values being imposed on the model";

  options.set_output("above_variable");
  options.set("above_variable").doc() =
      "The prescribed variable when the control signal is greater than the threshold";

  options.set_output("below_variable");
  options.set("below_variable").doc() =
      "The prescribed variable when the control signal is less than the threshold";

  return options;
}

MixedControlSetup::MixedControlSetup(const OptionSet & options)
  : Model(options),
    _threshold(declare_buffer<SR2>("c", "threshold")),
    _control(declare_input_variable<SR2>("control")),
    _fixed_values(declare_input_variable<SR2>("fixed_values")),
    _mixed_state(declare_input_variable<SR2>("mixed_state")),
    _above_var(declare_output_variable<SR2>("above_variable")),
    _below_var(declare_output_variable<SR2>("below_variable"))
{
}

void
MixedControlSetup::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto [dbelow, dabove] = make_operators(_control());

  if (out)
  {
    _above_var = dabove * _fixed_values + dbelow * _mixed_state;
    _below_var = dbelow * _fixed_values + dabove * _mixed_state;
  }

  if (dout_din)
  {
    if (_fixed_values.is_dependent())
    {
      _above_var.d(_fixed_values) = dabove;
      _below_var.d(_fixed_values) = dbelow;
    }

    if (_mixed_state.is_dependent())
    {
      _above_var.d(_mixed_state) = dbelow;
      _below_var.d(_mixed_state) = dabove;
    }
  }

  // All zero
  (void)(d2out_din2);
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
