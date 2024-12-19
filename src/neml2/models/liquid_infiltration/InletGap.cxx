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

#include "neml2/models/liquid_infiltration/InletGap.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(InletGap);
OptionSet
InletGap::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() = "The width of the RVE's inlet";

  options.set_parameter<CrossRef<Scalar>>("initial_porosity");
  options.set("initial_porosity").doc() =
      "Initial porosity of the RVE in the absence of product and liquid.";

  options.set_parameter<CrossRef<Scalar>>("product_thickness_growth_ratio");
  options.set("product_thickness_growth_ratio").doc() =
      "Volume expansion ratio towards the liquid from the product - solid reactions."
      "Value should be between 0 and 1. For Si + C -> SiC, set this to 0.576.";

  options.set_input("product_thickness") = VariableName("state", "delta");
  options.set("product_thickness").doc() = "Thickness of the product in the RVE.";

  options.set_output("inlet_gap") = VariableName("state", "r1");
  options.set("inlet_gap").doc() = "Width of the RVE's inlet.";

  return options;
}

InletGap::InletGap(const OptionSet & options)
  : Model(options),
    _M(declare_parameter<Scalar>("M",
                                 "product_thickness_growth_ratio")), //, /*allow_nonlinear=*/true)),
    _phi0(declare_parameter<Scalar>("phi0", "initial_porosity")),    //,/*allow_nonlinear=*/true)),
    _sqrtd(declare_input_variable<Scalar>("product_thickness")),
    _r1(declare_output_variable<Scalar>("inlet_gap"))
{
}

void
InletGap::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _r1 = math::sqrt(_phi0) - _M * _sqrtd * _sqrtd;
  }

  if (dout_din)
  {
    _r1.d(_sqrtd) = -2.0 * _M * _sqrtd;
  }
}
}