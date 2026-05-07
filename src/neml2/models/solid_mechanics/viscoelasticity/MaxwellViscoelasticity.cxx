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

#include "neml2/models/solid_mechanics/viscoelasticity/MaxwellViscoelasticity.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(MaxwellViscoelasticity);

OptionSet
MaxwellViscoelasticity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Dashpot evolution equation for a Maxwell viscoelastic element, i.e., \\f$ "
      "\\dot{\\boldsymbol{\\varepsilon}}_v = \\boldsymbol{\\sigma} / \\eta \\f$, where \\f$ \\eta "
      "\\f$ is the viscosity.";

  options.add_input("stress", "Stress acting across the dashpot");
  options.add_output("viscous_strain_rate",
                     "Rate of viscous strain. Override to match the rate name expected by the "
                     "time integrator if your state variable is not named `viscous_strain`.");
  options.add_parameter<Scalar>("viscosity", "Dashpot viscosity");

  return options;
}

MaxwellViscoelasticity::MaxwellViscoelasticity(const OptionSet & options)
  : Model(options),
    _S(declare_input_variable<SR2>("stress")),
    _Ev_dot(declare_output_variable<SR2>("viscous_strain_rate")),
    _eta(declare_parameter<Scalar>("eta", "viscosity", true))
{
}

void
MaxwellViscoelasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _Ev_dot = _S / _eta;

  if (dout_din)
  {
    _Ev_dot.d(_S) = imap_v<SR2>(_S.options()) / _eta;

    if (const auto * const eta = nl_param("eta"))
      _Ev_dot.d(*eta) = -_S() / (_eta * _eta);
  }
}
} // namespace neml2
