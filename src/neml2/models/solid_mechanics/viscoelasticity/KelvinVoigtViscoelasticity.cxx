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

#include "neml2/models/solid_mechanics/viscoelasticity/KelvinVoigtViscoelasticity.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(KelvinVoigtViscoelasticity);

OptionSet
KelvinVoigtViscoelasticity::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Stress response of a Kelvin-Voigt viscoelastic element (spring and dashpot in "
      "parallel), \\f$ \\boldsymbol{\\sigma} = E \\boldsymbol{\\varepsilon} + \\eta "
      "\\dot{\\boldsymbol{\\varepsilon}} \\f$, where \\f$ E \\f$ is the spring modulus and \\f$ "
      "\\eta \\f$ is the dashpot viscosity.";

  options.add_input("strain", "Strain shared by the spring and the dashpot");
  options.add_output("stress", "Stress in the Kelvin-Voigt element");
  options.add_parameter<Scalar>("modulus", "Spring modulus");
  options.add_parameter<Scalar>("viscosity", "Dashpot viscosity");

  return options;
}

KelvinVoigtViscoelasticity::KelvinVoigtViscoelasticity(const OptionSet & options)
  : Model(options),
    _E(declare_input_variable<SR2>("strain")),
    _E_dot(declare_input_variable<SR2>(rate_name(_E.name()))),
    _S(declare_output_variable<SR2>("stress")),
    _K(declare_parameter<Scalar>("K", "modulus", true)),
    _eta(declare_parameter<Scalar>("eta", "viscosity", true))
{
}

void
KelvinVoigtViscoelasticity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _S = _K * _E + _eta * _E_dot;

  if (dout_din)
  {
    auto I = imap_v<SR2>(_E.options());
    _S.d(_E) = _K * I;
    _S.d(_E_dot) = _eta * I;

    if (const auto * const K = nl_param("K"))
      _S.d(*K) = _E();

    if (const auto * const eta = nl_param("eta"))
      _S.d(*eta) = _E_dot();
  }
}
} // namespace neml2
