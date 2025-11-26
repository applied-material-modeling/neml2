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

#include "neml2/models/phase_field_fracture/RationalDegradationFunction.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(RationalDegradationFunction);

OptionSet
RationalDegradationFunction::expected_options()
{
  OptionSet options = DegradationFunction::expected_options();
  options.doc() =
      "Power degradation function to degrade the elastic strain energy density, \\f$ g "
      "= \\frac{\\left( 1-d \\right)^p}{\\left( 1-d \\right)^p + Q\\left(d \\right)} \\f$ where, "
      "\\f$ Q\\left(d \\right) = b_{1}d\\left( 1+b_{2}d+b_{2}b_{3}d^2 \\right)\\f$";
  options.set<TensorName<Scalar>>("power");
  options.set("power").doc() = "Power of the degradation function";

  options.set<double>("eta") = 0;
  options.set("eta").doc() = "Residual degradation when d = 1";

  options.set<TensorName<Scalar>>("fitting_param_1");
  options.set("fitting_param_1").doc() = "Material dependent fitting parameter 1";

  options.set<TensorName<Scalar>>("fitting_param_2");
  options.set("fitting_param_2").doc() = "Material dependent fitting parameter 2";

  options.set<TensorName<Scalar>>("fitting_param_3");
  options.set("fitting_param_3").doc() = "Material dependent fitting parameter 3";

  options.set<bool>("define_second_derivatives") = true;

  return options;
}

RationalDegradationFunction::RationalDegradationFunction(const OptionSet & options)
  : DegradationFunction(options),
    _p(declare_parameter<Scalar>("p", "power")),
    _eta(options.get<double>("eta")),
    _b1(declare_parameter<Scalar>("b1", "fitting_param_1")),
    _b2(declare_parameter<Scalar>("b2", "fitting_param_2")),
    _b3(declare_parameter<Scalar>("b3", "fitting_param_3"))

{
}

void
RationalDegradationFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto Q = _b1 * _d * (1 + _b2 * _d + _b2 * _b3 * _d * _d);
  if (out)
  {
    _g = pow((1 - _d), _p) / (pow((1 - _d), _p) + Q) * (1 - _eta) + _eta;
  }

  auto Q_prime = _b1 * (1 + 2 * _b2 * _d + 3 * _b2 * _b3 * _d * _d);
  auto u = _p * (pow((1 - _d), _p) + Q) * pow((1 - _d), (_p - 1)) * (-1.0) -
           pow((1 - _d), _p) * (_p * pow((1 - _d), (_p - 1)) * (-1.0) + Q_prime);
  auto v = (pow((1 - _d), _p) + Q) * (pow((1 - _d), _p) + Q);

  if (dout_din)
  {
    _g.d(_d) = (u / v) * (1 - _eta);
  }

  auto Q_double_prime = _b1 * (2 * _b2 + 6 * _b2 * _b3 * _d);
  auto u_prime =
      _p * ((_p * pow((1 - _d), (_p - 1)) * (-1.0) + Q_prime) * pow((1 - _d), (_p - 1)) * (-1.0) +
            (_p - 1) * pow((1 - _d), (_p - 2)) * (pow((1 - _d), _p) + Q)) -
      pow((1 - _d), _p) * (_p * (_p - 1) * pow((1 - _d), (_p - 2)) + Q_double_prime) -
      (_p * pow((1 - _d), (_p - 1)) * (-1.0) + Q_prime) * _p * pow((1 - _d), (_p - 1)) * (-1.0);
  auto v_prime = 2 * (pow((1 - _d), _p) + Q) * (_p * pow((1 - _d), (_p - 1)) * (-1.0) + Q_prime);

  if (d2out_din2)
  {
    _g.d2(_d, _d) = ((v * u_prime - u * v_prime) / (v * v)) * (1 - _eta);
  }
}
} // namespace neml2
