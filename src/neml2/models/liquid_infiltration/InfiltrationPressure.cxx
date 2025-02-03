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

#include "neml2/models/liquid_infiltration/InfiltrationPressure.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(InfiltrationPressure);
OptionSet
InfiltrationPressure::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Calculate the pressure from liquid's saturation, with the form of \\f$ "
                  "P0/(1-alpha_ratio) \\f$";

  options.set_parameter<CrossRef<Scalar>>("initial_pressure");
  options.set("initial_pressure").doc() = "Initial pressure P0";

  options.set_input("alpha_alphamax_ratio") = VariableName("state", "alpha_alphamax_ratio");
  options.set("alpha_alphamax_ratio").doc() =
      "The ratio between liquid's saturation and its maximum allowable saturation";

  options.set_output("pressure") = VariableName("state", "pressure");
  options.set("pressure").doc() = "The output pressure.";
  return options;
}

InfiltrationPressure::InfiltrationPressure(const OptionSet & options)
  : Model(options),
    _P0(declare_parameter<Scalar>("P0", "initial_pressure")),
    _alpha_rat(declare_input_variable<Scalar>("alpha_alphamax_ratio")),
    _P(declare_output_variable<Scalar>("pressure"))
{
}

void
InfiltrationPressure::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    // _P = _P0 / (1.0 - _alpha_rat + machine_precision());
    _P = _P0 * math::exp(-1.0 + _alpha_rat);
  }

  if (dout_din)
  {
    // _P.d(_alpha_rat) = _P0 / ((1.0 - _alpha_rat) * (1.0 - _alpha_rat) + machine_precision());
    _P.d(_alpha_rat) = _P0 * math::exp(-1.0 + _alpha_rat);
  }
}
}