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

#include "neml2/models/liquid_infiltration/PowerLawLiquidDeficiency.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(PowerLawLiquidDeficiency);
OptionSet
PowerLawLiquidDeficiency::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Scaling relations for product's thickness growth when the infiltrated liquid's "
      "span is less than the product's span. "
      "Current implementation assumed a simple power scale \\f$ \\frac{h_{L}}{h_{P}}^p \\f$";

  options.set_parameter<CrossRef<Scalar>>("exponent");
  options.set("exponent").doc() = "Exponent p in the model.";

  options.set_input("product_span") = VariableName("state", "hP");
  options.set("product_span").doc() = "The product's span.";

  options.set_input("liquid_span") = VariableName("state", "hL");
  options.set("liquid_span").doc() = "The liquid's span.";

  options.set_output("scale") = VariableName("state", "scale");
  options.set("scale").doc() = "Scaling relation of the model.";

  return options;
}

PowerLawLiquidDeficiency::PowerLawLiquidDeficiency(const OptionSet & options)
  : Model(options),
    _p(declare_parameter<Scalar>("p", "exponent")),
    _hL(declare_input_variable<Scalar>("liquid_span")),
    _hP(declare_input_variable<Scalar>("product_span")),
    _scale(declare_output_variable<Scalar>("scale"))
{
}

void
PowerLawLiquidDeficiency::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _scale = math::pow(math::macaulay(_hL * neml2::Scalar::full(1.0)) / _hP, _p);
  }

  if (dout_din)
  {
    _scale.d(_hP) =
        -(_p * math::pow(math::macaulay(_hL * neml2::Scalar::full(1.0)) / _hP, _p)) / _hP;
    _scale.d(_hL) = (_p * math::pow(math::macaulay(_hL * neml2::Scalar::full(1.0)) / _hP, _p)) /
                    (_hL + machine_precision());
  }
}
}