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

#include "neml2/models/liquid_infiltration/LiquidDeficientPowerScale.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LiquidDeficientPowerScale);
OptionSet
LiquidDeficientPowerScale::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Scaling relations for product's thickness growth when the infiltrated liquid's "
      "height is less than the product's height. "
      "Current implementation assumed a simple power scale \\f$ \\frac{h_{Si}}{h_{SiC}}^p \\f$";

  options.set_parameter<CrossRef<Scalar>>("liquid_molar_volume");
  options.set("liquid_molar_volume").doc() =
      "Infiltrated liquid's molar volume, units of molar mass per volume.";

  options.set_parameter<CrossRef<Scalar>>("power");
  options.set("power").doc() = "Power p in the model.";

  options.set_input("inlet_gap") = VariableName("state", "r1");
  options.set("inlet_gap").doc() = "The width of the inlet.";

  options.set_input("product_height") = VariableName("state", "h");
  options.set("product_height").doc() = "The product's height.";

  options.set_input("liquid_saturation") = VariableName("state", "alpha");
  options.set("liquid_saturation").doc() = "The current amount of the infiltrated liquid.";

  options.set_output("scale") = VariableName("state", "scale");
  options.set("scale").doc() = "Scaling relation of the model.";

  return options;
}

LiquidDeficientPowerScale::LiquidDeficientPowerScale(const OptionSet & options)
  : Model(options),
    _oL(declare_parameter<Scalar>("oL", "liquid_molar_volume", /*allow_nonlinear=*/true)),
    _p(declare_parameter<Scalar>("p", "power")),
    _r1(declare_input_variable<Scalar>("inlet_gap")),
    _h(declare_input_variable<Scalar>("product_height")),
    _alpha(declare_input_variable<Scalar>("liquid_saturation")),
    _scale(declare_output_variable<Scalar>("scale"))
{
}

void
LiquidDeficientPowerScale::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  auto hL = _alpha * _oL / (_r1 * _r1 + machine_precision());
  if (out)
  {
    _scale = math::pow(math::macaulay(hL) / _h, _p);
  }

  if (dout_din)
  {
    const auto * const oL = nl_param("oL");
    if (oL)
      _scale.d(*oL) = (_p * math::pow(math::macaulay(hL) / _h, _p)) / _oL;

    _scale.d(_r1) = -(2.0 * _p * math::pow(math::macaulay(hL) / _h, _p)) / _r1;
    _scale.d(_h) = -(_p * math::pow(math::macaulay(hL) / _h, _p)) / _h;
    _scale.d(_alpha) = (_p * math::pow(math::macaulay(hL) / _h, _p)) / _alpha;
  }
}
}