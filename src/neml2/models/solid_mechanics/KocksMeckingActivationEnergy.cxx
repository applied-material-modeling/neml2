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

#include "neml2/models/solid_mechanics/KocksMeckingActivationEnergy.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/log.h"

namespace neml2
{
register_NEML2_object(KocksMeckingActivationEnergy);

OptionSet
KocksMeckingActivationEnergy::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Calculates the Kocks-Mecking normalized activation as \\f$g = \\frac{kT}{\\mu "
      "b^3} \\log \\frac{\\dot{\\varepsilon}_0}{\\dot{\\varepsilon}} \\f$ with \\f$ "
      "\\mu \\f$ the shear modulus, \\f$ k \\f$ the Boltzmann constant, \\f$ T \\f$ the absolute "
      "temperature, \\f$ b \\f$ the Burgers vector length, \\f$ \\dot{\\varepsilon}_0 \\f$ a "
      "reference strain rate, and \\f$ \\dot{\\varepsilon} \\f$ the current strain rate.";

  options.add_parameter<Scalar>("shear_modulus", "The shear modulus");

  options.add<double>("eps0", "Reference strain rate");
  options.add<double>("k", "The Boltzmann constant");
  options.add<double>("b", "Magnitude of the Burgers vector");

  options.add_input("temperature", "Absolute temperature");
  options.add_input("strain_rate", "Effective strain rate");
  options.add_output("activation_energy", "Output name of the activation energy");
  return options;
}

KocksMeckingActivationEnergy::KocksMeckingActivationEnergy(const OptionSet & options)
  : Model(options),
    _mu(declare_parameter<Scalar>("mu", "shear_modulus", /*allow_nonlinear=*/true)),
    _eps0(options.get<double>("eps0")),
    _k(options.get<double>("k")),
    _b3(options.get<double>("b") * options.get<double>("b") * options.get<double>("b")),
    _T(declare_input_variable<Scalar>("temperature")),
    _eps_dot(declare_input_variable<Scalar>("strain_rate")),
    _g(declare_output_variable<Scalar>("activation_energy"))
{
}

void
KocksMeckingActivationEnergy::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _g = _k * _T / (_mu * _b3) * log(_eps0 / _eps_dot);

  if (dout_din)
  {
    _g.d(_T) = _k / (_mu * _b3) * log(_eps0 / _eps_dot);
    _g.d(_eps_dot) = -_k * _T / (_mu * _b3 * _eps_dot);

    if (const auto * const mu = nl_param("mu"))
      _g.d(*mu) = -_k * _T / (_b3 * _mu * _mu) * log(_eps0 / _eps_dot);
  }
}
} // namespace neml2
