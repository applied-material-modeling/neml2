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

#include "neml2/models/liquid_infiltration/LiquidSpan.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LiquidSpan);
OptionSet
LiquidSpan::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "The span of the liquid for the liquid infiltraion model";

  options.set_parameter<CrossRef<Scalar>>("liquid_molar_volume");
  options.set("liquid_molar_volume").doc() =
      "Infiltrated liquid's molar volume, units of molar mass per volume.";

  options.set_input("liquid_saturation") = VariableName("state", "alpha");
  options.set("liquid_saturation").doc() = "The current amount of the infiltrated liquid.";

  options.set_input("inlet_gap") = VariableName("state", "r1");
  options.set("inlet_gap").doc() = "The width of the inlet.";

  options.set_output("liquid_span") = VariableName("state", "scale");
  options.set("liquid_span").doc() = "The span of the liquid.";

  return options;
}

LiquidSpan::LiquidSpan(const OptionSet & options)
  : Model(options),
    _oL(declare_parameter<Scalar>("oL", "liquid_molar_volume", /*allow_nonlinear=*/true)),
    _alpha(declare_input_variable<Scalar>("liquid_saturation")),
    _r1(declare_input_variable<Scalar>("inlet_gap")),
    _hL(declare_output_variable<Scalar>("liquid_span"))
{
}

void
LiquidSpan::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _hL = _alpha * _oL / (_r1 * _r1 + machine_precision());
  }

  if (dout_din)
  {
    const auto * const oL = nl_param("oL");
    if (oL)
      _hL.d(*oL) = _alpha / (_r1 * _r1 + machine_precision());

    _hL.d(_alpha) = _oL / (_r1 * _r1 + machine_precision());
    _hL.d(_r1) = -2.0 * _alpha * _oL / (_r1 * _r1 * _r1 + machine_precision());
  };
}
}