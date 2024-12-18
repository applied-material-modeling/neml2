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

#include "neml2/models/liquid_infiltration/SwitchingFunction.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(SwitchingFunction);
OptionSet
SwitchingFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Smooth switching function of the form \\f$ 1 - \\sigmoid{x - 1, n} \\f$, "
                  "where n controls the sharpness of the transition.";

  options.set_parameter<CrossRef<Scalar>>("smooth_degree");
  options.set("smooth_degree").doc() = "n, sharpness of the transition.";

  options.set_input("variable") = VariableName("state", "var");
  options.set("variable").doc() = "x, input to the sigmoid function.";

  options.set_output("switch_out") = VariableName("state", "out");
  options.set("switch_out").doc() = "Smooth switching function.";

  return options;
}

SwitchingFunction::SwitchingFunction(const OptionSet & options)
  : Model(options),
    _nn(declare_parameter<Scalar>("nn", "smooth_degree")),
    _var(declare_input_variable<Scalar>("variable")),
    _smooth(declare_output_variable<Scalar>("switch_out"))
{
}

void
SwitchingFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _smooth = 1.0 - math::sigmoid(_var - 1.0, _nn);
  }

  if (dout_din)
  {
    _smooth.d(_var) =
        1.0 / 2.0 * (_nn * (math::tanh(_nn * (_var - 1.0)) * math::tanh(_nn * (_var - 1.0)) - 1.0));
  }
}
}