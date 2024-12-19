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

#include "neml2/models/liquid_infiltration/ProductThicknessGrowthRate.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductThicknessGrowthRate);
OptionSet
ProductThicknessGrowthRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "The residual associated with the thickness growth rate of the product";

  options.set_input("thickness_rate") = VariableName("state", "ddot");
  options.set("thickness_rate").doc() = "Product's thickness growth rate from the reaction";

  options.set_input("scale") = VariableName("state", "scale");
  options.set("scale").doc() =
      "Scaling relations for product's thickness growth when the infiltrated liquid's "
      "height is less than the product's height. ";

  options.set_input("ideal_thickness_growth") = VariableName("state", "rate");
  options.set("ideal_thickness_growth").doc() =
      "Product's thickness growth rate from the reaction in the abundance of infiltrated liquid";

  options.set_input("switch") = VariableName("state", "switch");
  options.set("switch").doc() = "Transition function to ensure the product thickness do not exceed "
                                "the maximum allowable thickness from the reaction ";

  options.set_output("residual_delta") = VariableName("residual", "rdelta");
  options.set("residual_delta").doc() =
      "Residual associated with the thickness growth rate of the product";

  return options;
}

ProductThicknessGrowthRate::ProductThicknessGrowthRate(const OptionSet & options)
  : Model(options),
    _ddot(declare_input_variable<Scalar>("thickness_rate")),
    _scale(declare_input_variable<Scalar>("scale")),
    _rate(declare_input_variable<Scalar>("ideal_thickness_growth")),
    _smooth(declare_input_variable<Scalar>("switch")),
    _rdelta(declare_output_variable<Scalar>("residual_delta"))
{
}

void
ProductThicknessGrowthRate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    // std::cout << "_ddot = " << _ddot.value().item<Real>() << std::endl;
    // std::cout << "_scale = " << _scale.value().item<Real>() << std::endl;
    // std::cout << "_rate = " << _rate.value().item<Real>() << std::endl;
    // std::cout << "_smooth = " << _smooth.value().item<Real>() << std::endl;

    _rdelta = _ddot - _scale * _rate * _smooth;
    // std::cout << "_rdelta = " << _rdelta.value().item<Real>() << std::endl;
  }

  if (dout_din)
  {
    _rdelta.d(_ddot) = neml2::Scalar::full(1.0);
    _rdelta.d(_scale) = -_rate * _smooth;
    _rdelta.d(_rate) = -_scale * _smooth;
    _rdelta.d(_smooth) = -_scale * _rate;
  }
}
}