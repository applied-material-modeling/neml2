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

#include "neml2/models/MixedControlConstraint.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/diagonalize.h"

namespace neml2
{
register_NEML2_object(MixedControlConstraint);

OptionSet
MixedControlConstraint::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Map the residual associated with a mixed control back to the unknown variables. "
                  "This object can be thought of as the inverse map of MixedControlSetup.";

  options.set_input("control") = VariableName(FORCES, "control");
  options.set("control").doc() = "The control signal.";

  options.set<TensorName<SR2>>("threshold") = TensorName<SR2>("0.5");
  options.set("threshold").doc() = "The threshold to switch between the two control";

  options.set_output("mixed_residual") = VariableName(STATE, "mixed_residual");
  options.set("mixed_residual").doc() =
      "The name of the mixed residual tensor. This holds the residuals for the conjugate "
      "values to those being controlled";

  options.set_output("below_residual");
  options.set("below_residual").doc() = "The residual corresponding to the variable to target when "
                                        "the control signal is below the threshold";

  return options;
}

MixedControlConstraint::MixedControlConstraint(const OptionSet & options)
  : Model(options),
    _threshold(declare_buffer<SR2>("c", "threshold")),
    _control(declare_input_variable<SR2>("control")),
    _mixed_residual(declare_output_variable<SR2>("mixed_residual")),
    _below_residual(declare_input_variable<SR2>("below_residual"))
{
}

void
MixedControlConstraint::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto dbelow = make_operator(_control());

  if (out)
    _mixed_residual = dbelow * _below_residual;

  if (dout_din)
    _mixed_residual.d(_below_residual) = dbelow;

  // All zero
  (void)(d2out_din2);
}

SSR4
MixedControlConstraint::make_operator(const SR2 & control) const
{
  auto below = control <= _threshold;

  // convert to matching dtype
  auto ones_below = below.to(control.options());
  auto dbelow = base_diagonalize(ones_below);

  return dbelow;
}

} // namespace neml2
