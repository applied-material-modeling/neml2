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

#include "neml2/models/reactive_infiltration/EffectiveSaturation.h"

namespace neml2
{
register_NEML2_object(EffectiveSaturation);
OptionSet
EffectiveSaturation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculate the effective saturation (volume fraction), takes the form of \\f$ S = "
      "\\frac{\\alpha \\Omega - S_r}{1-S_r} where "
      "\\f$ alpha \\f$ is the saturation (units of mol per volume), \\f$ Omega "
      "is the molar volume (units of mol per volume), and \\f$ "
      "S_r \\f$ is the residual liquid volume fraction";

  options.set_parameter<TensorName<Scalar>>("residual_saturation") = {TensorName<Scalar>("0")};
  options.set("residual_saturation").doc() = "Liquid's residual saturation";

  options.set_parameter<TensorName<Scalar>>("molar_volume");
  options.set("molar_volume").doc() = "Molar volume";

  options.set_input("saturation") = VariableName(FORCES, "saturation");
  options.set("saturation").doc() = "Flow species saturation";

  options.set_output("effective_saturation") = VariableName(STATE, "effective_saturation");
  options.set("effective_saturation").doc() = "Effective saturation";

  return options;
}

EffectiveSaturation::EffectiveSaturation(const OptionSet & options)
  : Model(options),
    _Sr(declare_parameter<Scalar>("Sr", "residual_saturation")),
    _omega(declare_parameter<Scalar>("Omega", "molar_volume")),
    _alpha(declare_input_variable<Scalar>("saturation")),
    _S(declare_output_variable<Scalar>("effective_saturation"))
{
}

void
EffectiveSaturation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _S = (_alpha * _omega - _Sr) / (1 - _Sr);
  }

  if (dout_din)
  {
    _S.d(_alpha) = _omega / (1 - _Sr);
  }
}
}