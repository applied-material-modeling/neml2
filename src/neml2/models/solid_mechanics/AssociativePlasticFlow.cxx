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

#include "neml2/models/solid_mechanics/AssociativePlasticFlow.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(AssociativePlasticFlow);

OptionSet
AssociativePlasticFlow::expected_options()
{
  OptionSet options = FlowRule::expected_options();
  options.doc() +=
      " This object calculates the rate of plastic strain following associative flow rule, i.e. "
      "\\f$ \\dot{\\boldsymbol{\\varepsilon}}_p = - \\dot{\\gamma} \\frac{\\partial f}{\\partial "
      "\\boldsymbol{M}} \\f$, where \\f$ \\dot{\\boldsymbol{\\varepsilon}}_p \\f$ is the plastic "
      "strain, \\f$ \\dot{\\gamma} \\f$ is the flow rate, \\f$ f \\f$ is the yield function, and "
      "\\f$ \\boldsymbol{M} \\f$ is the Mandel stress.";

  options.set_input("flow_direction") = VariableName(STATE, "internal", "NM");
  options.set("flow_direction").doc() = "Flow direction which can be calculated using Normality";

  options.set_output("plastic_strain_rate") = VariableName(STATE, "internal", "Ep_rate");
  options.set("plastic_strain_rate").doc() = "Rate of plastic strain";

  return options;
}

AssociativePlasticFlow::AssociativePlasticFlow(const OptionSet & options)
  : FlowRule(options),
    _NM(declare_input_variable<SR2>("flow_direction")),
    _Ep_dot(declare_output_variable<SR2>("plastic_strain_rate"))
{
}

void
AssociativePlasticFlow::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // For associative flow,
  // Ep_dot = gamma_dot * NM
  //     NM = df/dM

  if (out)
    _Ep_dot = _gamma_dot * _NM;

  if (dout_din)
  {
    auto I = imap_v<SR2>(_gamma_dot.options());

    if (_gamma_dot.is_dependent())
      _Ep_dot.d(_gamma_dot) = _NM();

    if (_NM.is_dependent())
      _Ep_dot.d(_NM) = _gamma_dot * I;
  }
}
} // namespace neml2
