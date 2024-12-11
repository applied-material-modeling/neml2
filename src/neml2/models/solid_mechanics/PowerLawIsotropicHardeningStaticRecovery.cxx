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

#include "neml2/models/solid_mechanics/PowerLawIsotropicHardeningStaticRecovery.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(PowerLawIsotropicHardeningStaticRecovery);

OptionSet
PowerLawIsotropicHardeningStaticRecovery::expected_options()
{
  OptionSet options = IsotropicHardeningStaticRecovery::expected_options();
  options.doc() = " This particular model implements a power law recovery of the type "
                  "\\f$ \\dot{k} = -\\left(\\frac{\\lVert k \\rVert}{\\tau}\\right)^{n-1} "
                  "\\frac{k}{\\tau} \\f$";

  options.set_parameter<CrossRef<Scalar>>("tau");
  options.set("tau").doc() = "Recovery rate";
  options.set_parameter<CrossRef<Scalar>>("n");
  options.set("n").doc() = "Recovery exponent";

  return options;
}

PowerLawIsotropicHardeningStaticRecovery::PowerLawIsotropicHardeningStaticRecovery(
    const OptionSet & options)
  : IsotropicHardeningStaticRecovery(options),
    _tau(declare_parameter<Scalar>("tau", "tau", true)),
    _n(declare_parameter<Scalar>("n", "n", true))
{
}

void
PowerLawIsotropicHardeningStaticRecovery::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(
      !d2out_din2,
      "PowerLawIsotropicHardeningStaticRecovery model doesn't implement second derivatives.");

  if (out)
    _h_dot = -math::pow(math::abs(Scalar(_h)) / _tau, _n - 1.0) * _h / _tau;

  if (dout_din)
  {
    if (_h.is_dependent())
      _h_dot.d(_h) = -_n * math::pow(math::abs(_h / _tau), _n - 1) / math::abs(_tau);

    if (const auto * const tau = nl_param("tau"))
      _h_dot.d(*tau) =
          _n * _h * math::pow(_tau, -1 - _n) * math::pow(math::abs(Scalar(_h)), _n - 1);

    if (const auto * const n = nl_param("n"))
      _h_dot.d(*n) = -_h * math::pow(_tau, -_n) * math::pow(math::abs(Scalar(_h)), _n - 1) *
                     math::log(math::abs(Scalar(_h) + machine_precision()) / _tau);
  }
}
} // namespace neml2
