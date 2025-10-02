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

#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
register_NEML2_object(VoceIsotropicHardening);

OptionSet
VoceIsotropicHardening::expected_options()
{
  OptionSet options = IsotropicHardening::expected_options();
  options.doc() =
      "Voce isotropic hardening model, \\f$ h = R \\left[ 1 - \\exp(-d \\bar{\\varepsilon}_p) "
      "\\right] \\f$, where \\f$ R \\f$ is the isotropic hardening upon saturation, "
      "and \\f$ d \\f$ is the hardening rate.";

  options.set<bool>("define_second_derivatives") = true;

  options.set_parameter<TensorName<Scalar>>("saturated_hardening");
  options.set("saturated_hardening").doc() = "Saturated isotropic hardening";
  options.set_parameter<TensorName<Scalar>>("saturation_rate");
  options.set("saturation_rate").doc() = "Hardening saturation rate";

  return options;
}

VoceIsotropicHardening::VoceIsotropicHardening(const OptionSet & options)
  : IsotropicHardening(options),
    _R(declare_parameter<Scalar>("R", "saturated_hardening", true)),
    _d(declare_parameter<Scalar>("d", "saturation_rate", true))
{
}

void
VoceIsotropicHardening::set_value(bool out, bool dout_din, bool d2out_din2)
{
  if (out)
    _h = _R * (-exp(-_d * _ep) + 1.0);

  if (dout_din)
  {
    if (_ep.is_dependent())
      _h.d(_ep) = _R * _d * exp(-_d * _ep);

    if (const auto * const R = nl_param("R"))
      _h.d(*R) = -exp(-_d * _ep) + 1.0;

    if (const auto * const d = nl_param("d"))
      _h.d(*d) = _ep * _R * exp(-_d * _ep);
  }

  if (d2out_din2)
    if (_ep.is_dependent())
      _h.d(_ep, _ep) = -_R * _d * _d * exp(-_d * _ep);
}
} // namespace neml2
