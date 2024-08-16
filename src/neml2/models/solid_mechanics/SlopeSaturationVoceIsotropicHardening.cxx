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

#include "neml2/models/solid_mechanics/SlopeSaturationVoceIsotropicHardening.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(SlopeSaturationVoceIsotropicHardening);

OptionSet
SlopeSaturationVoceIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.doc() = "SlopeSaturationVoce isotropic hardening model, \\f$ h = R \\left[ 1 - "
                  "\\exp(-\\theta_0 \\varepsilon_p / R) "
                  "\\right] \\f$, where \\f$ R \\f$ is the isotropic hardening upon saturation, "
                  "and \\f$ \\theta_0 \\f$ is the initial hardening rate.";

  options.set_parameter<CrossRef<Scalar>>("saturated_hardening");
  options.set("saturated_hardening").doc() = "Saturated isotropic hardening";
  options.set_parameter<CrossRef<Scalar>>("initial_hardening_rate");
  options.set("initial_hardening_rate").doc() = "Initial hardening rate";

  return options;
}

SlopeSaturationVoceIsotropicHardening::SlopeSaturationVoceIsotropicHardening(
    const OptionSet & options)
  : IsotropicHardening(options),
    _R(declare_parameter<Scalar>("R", "saturated_hardening", true)),
    _theta0(declare_parameter<Scalar>("theta0", "initial_hardening_rate", true))
{
}

void
SlopeSaturationVoceIsotropicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _h = _R * (-math::exp(-_theta0 / _R * _ep) + 1.0);

  if (dout_din)
  {
    if (_ep.is_dependent())
      _h.d(_ep) = _theta0 * math::exp(-_theta0 / _R * _ep);

    if (const auto * const R = nl_param("R"))
      _h.d(*R) = 1.0 - (math::exp(-_theta0 * _ep / _R) * (_theta0 * _ep + _R)) / _R;

    if (const auto * const theta0 = nl_param("theta0"))
      _h.d(*theta0) = _ep * math::exp(-_theta0 * _ep / _R);
  }

  if (d2out_din2)
    if (_ep.is_dependent())
      _h.d(_ep, _ep) = -_theta0 * _theta0 * math::exp(-_theta0 / _R * _ep) / _R;
}
} // namespace neml2
