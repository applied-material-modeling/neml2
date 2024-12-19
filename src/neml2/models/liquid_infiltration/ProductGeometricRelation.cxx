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

#include "neml2/models/liquid_infiltration/ProductGeometricRelation.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ProductGeometricRelation);
OptionSet
ProductGeometricRelation::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Relationship between the product height, width and its saturation in mol/V.";

  options.set_parameter<CrossRef<Scalar>>("product_molar_volume");
  options.set("product_molar_volume").doc() = "Product's molar volume (mol/V).";

  options.set_input("inlet_gap") = VariableName("state", "r1");
  options.set("inlet_gap").doc() = "The width of the RVE's inlet.";

  options.set_input("product_height") = VariableName("state", "h");
  options.set("product_height").doc() = "Height of the product in the RVE.";

  options.set_input("product_thickness") = VariableName("state", "delta");
  options.set("product_thickness").doc() = "Thickness of the product in the RVE.";

  options.set_output("product_saturation") = VariableName("state", "alphaP");
  options.set("product_saturation").doc() = "The saturation of the product in mol/V.";

  return options;
}

ProductGeometricRelation::ProductGeometricRelation(const OptionSet & options)
  : Model(options),
    _oP(declare_parameter<Scalar>("oP", "product_molar_volume", /*allow_nonlinear=*/true)),
    _r1(declare_input_variable<Scalar>("inlet_gap")),
    _sqrtd(declare_input_variable<Scalar>("product_thickness")),
    _h(declare_input_variable<Scalar>("product_height")),
    _alphaP(declare_output_variable<Scalar>("product_saturation"))
{
}

void
ProductGeometricRelation::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _alphaP =
        _h / _oP *
        (2.0 * _r1 * _sqrtd * _sqrtd +
         _sqrtd * _sqrtd * _sqrtd * _sqrtd); // can not get power to work, math::pow(_sqrtd, 4.0));
  }

  if (dout_din)
  {
    const auto * const oP = nl_param("oP");
    if (oP)
      _alphaP.d(*oP) =
          -_h / (_oP * _oP) * (2.0 * _r1 * _sqrtd * _sqrtd + _sqrtd * _sqrtd * _sqrtd * _sqrtd);

    _alphaP.d(_r1) = 2.0 * _h * _sqrtd * _sqrtd / _oP;
    _alphaP.d(_h) = (2.0 * _r1 * _sqrtd * _sqrtd + _sqrtd * _sqrtd * _sqrtd * _sqrtd) / _oP;
    _alphaP.d(_sqrtd) = (4 * _sqrtd * _h * (_sqrtd * _sqrtd + _r1)) / _oP;
  }
}
}