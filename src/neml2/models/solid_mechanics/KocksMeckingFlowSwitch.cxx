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

#include "neml2/models/solid_mechanics/KocksMeckingFlowSwitch.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/tanh.h"
#include "neml2/tensors/functions/cosh.h"

namespace neml2
{
register_NEML2_object(KocksMeckingFlowSwitch);

OptionSet
KocksMeckingFlowSwitch::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Switches between rate independent and rate dependent flow rules based on the "
                  "value of the Kocks-Mecking normalized activation energy.  For activation "
                  "energies less than the threshold use the rate independent flow rule, for values "
                  "greater than the threshold use the rate dependent flow rule.  This version uses "
                  "a soft switch between the models, based on a tanh sigmoid function.";

  options.add_parameter<Scalar>("g0", "Critical value of activation energy");
  options.add_input("activation_energy", "The input name of the activation energy");
  options.add<double>(
      "sharpness",
      1.0,
      "A steepness parameter that controls the tanh mixing of the models.  Higher values gives a "
      "sharper transition.");

  options.add_input("rate_independent_flow_rate", "Input name of the rate independent flow rate");
  options.add_input("rate_dependent_flow_rate", "Input name of the rate dependent flow rate");
  options.add_output("flow_rate", "Output name for the mixed flow rate");

  return options;
}

KocksMeckingFlowSwitch::KocksMeckingFlowSwitch(const OptionSet & options)
  : Model(options),
    _g0(declare_parameter<Scalar>("g0", "g0", /*allow_nonlinear=*/true)),
    _g(declare_input_variable<Scalar>("activation_energy")),
    _sharp(options.get<double>("sharpness")),
    _ri_flow(declare_input_variable<Scalar>("rate_independent_flow_rate")),
    _rd_flow(declare_input_variable<Scalar>("rate_dependent_flow_rate")),
    _gamma_dot(declare_output_variable<Scalar>("flow_rate"))
{
}

void
KocksMeckingFlowSwitch::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto sig = (tanh(_sharp * (_g - _g0)) + 1.0) / 2.0;

  if (out)
    _gamma_dot = sig * _rd_flow + (1.0 - sig) * _ri_flow;

  if (dout_din)
  {
    _gamma_dot.d(_rd_flow) = sig;
    _gamma_dot.d(_ri_flow) = 1.0 - sig;

    auto partial = 0.5 * _sharp * pow(1.0 / cosh(_sharp * (_g - _g0)), 2.0);
    auto deriv = partial * (_rd_flow - _ri_flow);

    _gamma_dot.d(_g) = deriv;

    if (const auto * const g0 = nl_param("g0"))
      _gamma_dot.d(*g0) = -deriv;
  }
}

} // namespace neml2
