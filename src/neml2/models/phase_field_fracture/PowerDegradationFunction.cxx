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

#include "neml2/models/phase_field_fracture/PowerDegradationFunction.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(PowerDegradationFunction);

OptionSet
PowerDegradationFunction::expected_options()
{
  OptionSet options = DegradationFunction::expected_options();
  options.doc() = "Power degradation function to degrade the elastic strain energy density, \\f$ g "
                  "= \\left( 1-d \\right)^2 (1-\\eta) + \\eta \\f$";
  options.set<TensorName<Scalar>>("power");
  options.set("power").doc() = "Power of the degradation function";
  options.set<double>("eta") = 0;
  options.set("eta").doc() = "Residual degradation when d = 1";

  options.set<bool>("define_second_derivatives") = true;

  return options;
}

PowerDegradationFunction::PowerDegradationFunction(const OptionSet & options)
  : DegradationFunction(options),
    _p(declare_parameter<Scalar>("p", "power")),
    _eta(options.get<double>("eta"))
{
}

void
PowerDegradationFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
  {
    _g = pow((1 - _d), _p) * (1 - _eta) + _eta;
  }

  if (dout_din)
  {
    _g.d(_d) = -_p * pow((1 - _d), (_p - 1)) * (1 - _eta);
  }

  if (d2out_din2)
  {
    _g.d(_d, _d) = (_p * (_p - 1)) * pow((1 - _d), (_p - 2)) * (1 - _eta);
  }
}
} // namespace neml2
