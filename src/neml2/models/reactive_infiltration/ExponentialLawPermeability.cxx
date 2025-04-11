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

#include "neml2/models/reactive_infiltration/ExponentialLawPermeability.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(ExponentialLawPermeability);
OptionSet
ExponentialLawPermeability::expected_options()
{
  OptionSet options = PorosityPermeabilityRelation::expected_options();
  options.doc() =
      "Define the exponential porosity-permeability relation, takes the form of "
      "\\f$ K_o \\exp(a(\\varphi_o-\\varphi))} \\f$ where \\f$ a "
      "\\f$ is the fitting parameter. \\f$ varphi_o, K_o \\f$ are the reference porosity and "
      "permeability respectively.";

  options.set_parameter<TensorName<Scalar>>("exponential_scale");
  options.set("exponential_scale").doc() = "value of the exponential constant scale";

  return options;
}

ExponentialLawPermeability::ExponentialLawPermeability(const OptionSet & options)
  : PorosityPermeabilityRelation(options),
    _a(declare_parameter<Scalar>("a", "exponential_scale"))
{
}

void
ExponentialLawPermeability::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _K = _Ko * exp(_a * (_phio - _phi));
  }

  if (dout_din)
  {
    _K.d(_phi) = _Ko * exp(_a * (_phio - _phi)) * -_a;
  }
}
}