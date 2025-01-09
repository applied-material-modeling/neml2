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

#include "neml2/models/solid_mechanics/AssociativeKinematicPlasticHardening.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(AssociativeKinematicPlasticHardening);

OptionSet
AssociativeKinematicPlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() +=
      " This object calculates the rate of kinematic plastic strain following associative flow "
      "rule, i.e. \\f$ \\dot{\\boldsymbol{K}}_p = - \\dot{\\gamma} \\frac{\\partial f}{\\partial "
      "\\boldsymbol{X}} \\f$, where \\f$ \\dot{\\boldsymbol{K}}_p \\f$ is the kinematic plastic "
      "strain, \\f$ \\dot{\\gamma} \\f$ is the flow rate, \\f$ f \\f$ is the yield function, and "
      "\\f$ \\boldsymbol{X} \\f$ is the kinematic hardening.";

  options.set_input("kinematic_hardening_direction") = VariableName(STATE, "internal", "NX");
  options.set("kinematic_hardening_direction").doc() =
      "Direction of associative kinematic hardening which can be calculated using Normality.";

  options.set_output("kinematic_plastic_strain_rate") = VariableName(STATE, "internal", "Kp_rate");
  options.set("kinematic_plastic_strain_rate").doc() = "Rate of kinematic plastic strain";

  return options;
}

AssociativeKinematicPlasticHardening::AssociativeKinematicPlasticHardening(
    const OptionSet & options)
  : FlowRule(options),
    _NX(declare_input_variable<SR2>("kinematic_hardening_direction")),
    _Kp_dot(declare_output_variable<SR2>("kinematic_plastic_strain_rate"))
{
}

void
AssociativeKinematicPlasticHardening::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // For associative flow,
  // Kp_dot = - gamma_dot * NX
  //     NX = df/dX

  if (out)
    _Kp_dot = -_gamma_dot * _NX;

  if (dout_din)
  {
    auto I = SR2::identity_map(_gamma_dot.options());

    if (_gamma_dot.is_dependent())
      _Kp_dot.d(_gamma_dot) = -_NX;

    if (_NX.is_dependent())
      _Kp_dot.d(_NX) = -_gamma_dot * I;
  }
}
} // namespace neml2
