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

#include "neml2/models/reactive_infiltration/BrooksCoreyPressure.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/log10.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
register_NEML2_object(BrooksCoreyPressure);
OptionSet
BrooksCoreyPressure::expected_options()
{
  OptionSet options = PorousFlowCapillaryPressure::expected_options();
  options.doc() =
      "Define the Brooks Corey porous flow capillary pressure, takes the form of \\f$ "
      "P_c = P_t S_e^{-\\frac{1}{p}} \\f$ "
      "and 0 everywhere else. Here \\f$ S_e \\f$ is the effective saturation,\\f$ Pt, p "
      "\\f$ are the threshold pressure at zero saturation and the fitting parameter";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("threshold_pressure");
  options.set("threshold_pressure").doc() = "threshold entry pressure";

  options.set_parameter<TensorName<Scalar>>("power");
  options.set("power").doc() = "power, p";

  options.set<bool>("apply_log_extension") = false;
  options.set("apply_log_extension").doc() = "Whether to apply_log_extension";

  options.set<double>("transistion_saturation") = 0.1;
  options.set("transistion_saturation").doc() = "The transistion value of the effective saturation";

  return options;
}

BrooksCoreyPressure::BrooksCoreyPressure(const OptionSet & options)
  : PorousFlowCapillaryPressure(options),
    _Pt(declare_parameter<Scalar>("threshold", "threshold_pressure")),
    _p(declare_parameter<Scalar>("p", "power")),
    _log_extension(options.get<bool>("apply_log_extension")),
    _Sp(options.get<double>("transistion_saturation"))
{
}

void
BrooksCoreyPressure::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(_Sp < 1.0, "transistion_saturation cannot be larger or equal to 1");

  constexpr double ln10 = 2.302585092994;

  // required information for any model
  auto f_s = _Pt * pow(_S, -1.0 / _p);
  auto f_sp = _Pt * pow(_Sp, -1.0 / _p);

  auto dfds_s = -_Pt / _p * pow(_S, -1.0 / _p - 1.0);
  auto dfds_sp = -_Pt / _p * pow(_Sp, -1.0 / _p - 1.0);

  auto d2fds2_s = -_Pt / _p * (-1.0 / _p - 1.0) * pow(_S, -1.0 / _p - 2.0);

  // general for any given model
  auto log10f_s = log10(f_s);
  auto log10f_sp = log10(f_sp);

  // auto dlog10fds_sp = 1 / (ln10 * f_sp) * dfds_sp; // clang tidy throw this error ...

  auto slope = 1 / (ln10 * f_sp) * dfds_sp;
  auto yintercept = log10f_sp - slope * _Sp;

  if (out)
  {
    if (_log_extension)
      _Pc = where(_S < _Sp, pow(10, slope * _S + yintercept), f_s);
    else
      _Pc = f_s;
  }

  if (dout_din)
  {
    if (_log_extension)
      _Pc.d(_S) = where(_S < _Sp, (ln10 * slope) * pow(10, slope * _S + yintercept), dfds_s);
    else
      _Pc.d(_S) = dfds_s;
  }

  if (d2out_din2)
  {
    if (_log_extension)
      _Pc.d(_S, _S) =
          where(_S < _Sp, pow((ln10 * slope), 2) * pow(10, slope * _S + yintercept), d2fds2_s);
    else
      _Pc.d(_S, _S) = d2fds2_s;
  }
}
}
