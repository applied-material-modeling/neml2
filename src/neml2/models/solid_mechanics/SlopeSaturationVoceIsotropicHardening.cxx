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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sign.h"

namespace neml2
{
register_NEML2_object(SlopeSaturationVoceIsotropicHardening);

OptionSet
SlopeSaturationVoceIsotropicHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() = "SlopeSaturationVoce isotropic hardening model, \\f$ \\dot{h} = \\theta_0 "
                  "\\left(1 - \\frac{h}{R} \\right) \\varepsilon_p \\f$, where \\f$ R \\f$ is the "
                  "isotropic hardening upon saturation, "
                  "and \\f$ \\theta_0 \\f$ is the initial hardening rate. "
                  "In addition to the reparameterization, this model differences from the "
                  "`VoceIsotropicHardening` model in that it defines the hardening rate in a "
                  "non-assocative manner.  This is sometimes handy, for example in supplementing "
                  "the model with static recovery.";

  options.set_input("isotropic_hardening") = VariableName(STATE, "internal", "k");
  options.set("isotropic_hardening").doc() = "Isotropic hardening variable";

  options.set_input("isotropic_hardening_rate");
  options.set("isotropic_hardening_rate").doc() =
      "Rate of isotropic hardening, defaults to isotropic_hardening + _rate";

  options.set_parameter<TensorName>("saturated_hardening");
  options.set("saturated_hardening").doc() = "Saturated isotropic hardening";
  options.set_parameter<TensorName>("initial_hardening_rate");
  options.set("initial_hardening_rate").doc() = "Initial hardening rate";

  return options;
}

SlopeSaturationVoceIsotropicHardening::SlopeSaturationVoceIsotropicHardening(
    const OptionSet & options)
  : FlowRule(options),
    _h(declare_input_variable<Scalar>("isotropic_hardening")),
    _h_dot(declare_output_variable<Scalar>(
        options.get<VariableName>("isotropic_hardening_rate").empty()
            ? _h.name().with_suffix("_rate")
            : options.get<VariableName>("isotropic_hardening_rate"))),
    _R(declare_parameter<Scalar>("R", "saturated_hardening", true)),
    _theta0(declare_parameter<Scalar>("theta0", "initial_hardening_rate", true))
{
}

void
SlopeSaturationVoceIsotropicHardening::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _h_dot = sign(_R) * _theta0 * (1.0 - _h / _R) * _gamma_dot;

  if (dout_din)
  {
    if (_gamma_dot.is_dependent())
      _h_dot.d(_gamma_dot) = sign(_R) * _theta0 * (1.0 - _h / _R);

    if (_h.is_dependent())
      _h_dot.d(_h) = -sign(_R) * _theta0 / _R * _gamma_dot;

    if (const auto * const R = nl_param("R"))
      _h_dot.d(*R) = _gamma_dot * _h * sign(_R) * _theta0 / (_R * _R);

    if (const auto * const theta0 = nl_param("theta0"))
      _h_dot.d(*theta0) = sign(_R) * (1.0 - _h / _R) * _gamma_dot;
  }
}
} // namespace neml2
