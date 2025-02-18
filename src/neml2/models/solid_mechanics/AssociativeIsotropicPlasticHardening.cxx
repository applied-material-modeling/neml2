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

#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(AssociativeIsotropicPlasticHardening);

OptionSet
AssociativeIsotropicPlasticHardening::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() += " This object calculates the rate of equivalent plastic strain following "
                   "associative flow rule, i.e. \\f$ \\dot{\\varepsilon}_p = - \\dot{\\gamma} "
                   "\\frac{\\partial f}{\\partial k} \\f$, where \\f$ \\dot{\\varepsilon}_p \\f$ "
                   "is the equivalent plastic strain, \\f$ \\dot{\\gamma} \\f$ is the flow rate, "
                   "\\f$ f \\f$ is the yield function, and \\f$ k \\f$ is the isotropic hardening.";

  options.set_input("isotropic_hardening_direction") = VariableName(STATE, "internal", "Nk");
  options.set("isotropic_hardening_direction").doc() =
      "Direction of associative isotropic hardening which can be calculated using Normality.";

  options.set_output("equivalent_plastic_strain_rate") = VariableName(STATE, "internal", "ep_rate");
  options.set("equivalent_plastic_strain_rate").doc() = "Rate of equivalent plastic strain";

  return options;
}

AssociativeIsotropicPlasticHardening::AssociativeIsotropicPlasticHardening(
    const OptionSet & options)
  : FlowRule(options),
    _Nk(declare_input_variable<Scalar>("isotropic_hardening_direction")),
    _ep_dot(declare_output_variable<Scalar>("equivalent_plastic_strain_rate"))
{
}

void
AssociativeIsotropicPlasticHardening::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // For associative flow,
  // ep_dot = - gamma_dot * Nk
  //     Nk = df/dk

  if (out)
    _ep_dot = -_gamma_dot * _Nk;

  if (dout_din)
  {
    if (_gamma_dot.is_dependent())
      _ep_dot.d(_gamma_dot) = -_Nk;

    if (_Nk.is_dependent())
      _ep_dot.d(_Nk) = -_gamma_dot;
  }
}
} // namespace neml2
