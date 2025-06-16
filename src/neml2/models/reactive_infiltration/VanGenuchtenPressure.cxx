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

#include "neml2/models/reactive_infiltration/VanGenuchtenPressure.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/log10.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(VanGenuchtenPressure);
OptionSet
VanGenuchtenPressure::expected_options()
{
  OptionSet options = PorousFlowCapillaryPressure::expected_options();
  options.doc() = "Define the van Genuchten porous flow capillary pressure, takes the form of \\f$ "
                  "\\frac{1}{a} "
                  "\\frac{S_e^{-\\frac{1}{m}} - 1}{1-m} \\f$ for \\f$ S_e < 1 \\f$ "
                  "and 0 everywhere else. Here \\f$ S_e \\f$ is the effective saturation,\\f$ a, m "
                  "\\f$ is the fitting parameter";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("scaling_constant");
  options.set("scaling_constant").doc() = "scaling reciprocal constant, a";

  options.set_parameter<TensorName<Scalar>>("power");
  options.set("power").doc() = "power, m";

  options.set<bool>("apply_log_extension") = false;
  options.set("apply_log_extension").doc() = "Whether to apply_log_extension";

  options.set<double>("transistion_saturation") = 0.1;
  options.set("transistion_saturation").doc() = "The transistion value of the effective saturation";

  return options;
}

VanGenuchtenPressure::VanGenuchtenPressure(const OptionSet & options)
  : PorousFlowCapillaryPressure(options),
    _a(declare_parameter<Scalar>("a", "scaling_constant")),
    _m(declare_parameter<Scalar>("m", "power")),
    _log_extension(options.get<bool>("apply_log_extension")),
    _Sp(options.get<double>("transistion_saturation"))
{
}

void
VanGenuchtenPressure::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(_Sp < 1.0, "transistion_saturation cannot be larger or equal to 1");

  constexpr double ln10 = 2.302585092994;

  const auto eps = machine_precision(_S.scalar_type()).toDouble();
  auto _Sclamp = clamp(_S, 0.0, 1.0 - eps);

  // required information for any model
  auto f_s = 1 / _a * pow((pow(_Sclamp, -1.0 / _m) - 1.0), 1 - _m);
  auto f_sp = 1 / _a * pow((pow(_Sp, -1.0 / _m) - 1.0), 1 - _m);

  auto dfds_s =
      -1 / (_a * _m) * (1 - _m) * pow(pow(_Sclamp, -1 / _m) - 1, -_m) * pow(_Sclamp, -1 / _m - 1);
  auto dfds_sp =
      -1 / (_a * _m) * (1 - _m) * pow(pow(_Sp, -1 / _m) - 1, -_m) * pow(_Sp, -1 / _m - 1);

  auto d2fds2_s = -((_m - 1) * (_m * pow(_Sclamp, (1 / _m)) + pow(_Sclamp, (1 / _m)) - 1)) /
                  (_a * _m * _m * pow(_Sclamp, (1 / _m + 2)) *
                   pow((1 / pow(_Sclamp, (1 / _m)) - 1), _m) * (pow(_Sclamp, (1 / _m)) - 1));

  // general for any given model
  auto log10f_s = log10(f_s);
  auto log10f_sp = log10(f_sp);

  // auto dlog10fds_sp = 1 / (ln10 * f_sp) * dfds_sp;

  auto slope = 1 / (ln10 * f_sp) * dfds_sp;
  auto yintercept = log10f_sp - slope * _Sp;

  if (out)
  {
    if (_log_extension)
      _Pc = where(_S < _Sp, pow(10, slope * _Sclamp + yintercept), f_s);
    else
      _Pc = f_s;
  }

  if (dout_din)
  {
    if (_log_extension)
      _Pc.d(_S) = where(_S < _Sp, (ln10 * slope) * pow(10, slope * _Sclamp + yintercept), dfds_s);
    else
      _Pc.d(_S) = dfds_s;
  }

  if (d2out_din2)
  {
    if (_log_extension)
      _Pc.d(_S, _S) =
          where(_S < _Sp, pow((ln10 * slope), 2) * pow(10, slope * _Sclamp + yintercept), d2fds2_s);
    else
      _Pc.d(_S, _S) = d2fds2_s;
  }
}
}
