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

#include "neml2/models/solid_mechanics/PowerLawKinematicHardeningStaticRecovery.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(PowerLawKinematicHardeningStaticRecovery);

OptionSet
PowerLawKinematicHardeningStaticRecovery::expected_options()
{
  OptionSet options = KinematicHardeningStaticRecovery::expected_options();
  options.doc() +=
      " This particular model uses a power law for recovery "
      "\\f$ \\dot{X} = - \\left(\\frac{\\lVert X \\rVert}{\\tau}\\right)^{n-1} \\frac{X}{\\tau} "
      "\\f$"
      "where \\f$ n \\f$ is the power law recovery exponent and \\f$\\tau\\f$ is the recovery "
      "rate.";

  options.set_parameter<CrossRef<Scalar>>("tau");
  options.set("tau").doc() = "Static recovery rate";

  options.set_parameter<CrossRef<Scalar>>("n");
  options.set("n").doc() = "Static recovery exponent";

  return options;
}

PowerLawKinematicHardeningStaticRecovery::PowerLawKinematicHardeningStaticRecovery(
    const OptionSet & options)
  : KinematicHardeningStaticRecovery(options),
    _tau(declare_parameter<Scalar>("tau", "tau", true)),
    _n(declare_parameter<Scalar>("n", "n", true))
{
}

void
PowerLawKinematicHardeningStaticRecovery::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(
      !d2out_din2,
      "PowerLawKinematicHardeningStaticRecovery model doesn't implement second derivatives.");

  // The effective stress
  auto s = SR2(_X).norm(machine_precision());

  if (out)
    _X_dot = -math::pow(s / _tau, _n - 1) * _X / _tau;

  if (dout_din)
  {
    auto I = SR2::identity_map(options());

    if (_X.is_dependent())
      _X_dot.d(_X) = -math::pow(s, _n - 3) * ((_n - 1) * SR2(_X).outer(SR2(_X)) + s * s * I) /
                     math::pow(_tau, _n);

    if (const auto * const tau = nl_param("tau"))
      _X_dot.d(*tau) = _n * math::pow(s / _tau, _n - 1) * _X / (_tau * _tau);

    if (const auto * const n = nl_param("n"))
      _X_dot.d(*n) = -_X / s * math::pow(s / _tau, _n) * math::log((s+machine_precision()) / _tau);
  }
}

} // namespace neml2
