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

#include "neml2/models/kwn/NuceationFluxMagnitude.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
register_NEML2_object(NuceationFluxMagnitude);

OptionSet
NuceationFluxMagnitude::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the nucleation flux magnitude excluding the Dirac delta term.";

  options.set_input("zeldovich_factor");
  options.set("zeldovich_factor").doc() = "Zeldovich factor";

  options.set_input("kinetic_factor");
  options.set("kinetic_factor").doc() = "Kinetic factor";

  options.set_input("nucleation_barrier");
  options.set("nucleation_barrier").doc() = "Nucleation barrier";

  options.set_parameter<TensorName<Scalar>>("temperature");
  options.set("temperature").doc() = "Temperature";

  options.set_parameter<TensorName<Scalar>>("nucleation_site_density");
  options.set("nucleation_site_density").doc() = "Nucleation site density";

  options.set_parameter<TensorName<Scalar>>("boltzmann_constant");
  options.set("boltzmann_constant").doc() = "Boltzmann constant";

  options.set_output("nucleation_flux_magnitude");
  options.set("nucleation_flux_magnitude").doc() =
      "Nucleation flux magnitude excluding the Dirac delta term";

  return options;
}

NuceationFluxMagnitude::NuceationFluxMagnitude(const OptionSet & options)
  : Model(options),
    _Z(declare_input_variable<Scalar>("zeldovich_factor")),
    _beta(declare_input_variable<Scalar>("kinetic_factor")),
    _dg(declare_input_variable<Scalar>("nucleation_barrier")),
    _T(declare_parameter<Scalar>("T", "temperature", /*allow_nonlinear=*/true)),
    _N0(declare_parameter<Scalar>("N0", "nucleation_site_density", /*allow_nonlinear=*/true)),
    _k(declare_parameter<Scalar>("k", "boltzmann_constant")),
    _J(declare_output_variable<Scalar>("nucleation_flux_magnitude"))
{
}

void
NuceationFluxMagnitude::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto Z = _Z();
  const auto beta = _beta();
  const auto dg = _dg();
  const auto T = _T;
  const auto N0 = _N0;
  const auto k = _k;

  const auto arg = -dg / (k * T);
  const auto exp_raw = exp(arg);

  const auto prefactor = Z * beta * N0;
  const auto J = prefactor * exp_raw;

  if (out)
    _J = J;

  if (dout_din)
  {
    if (_Z.is_dependent())
      _J.d(_Z) = beta * N0 * exp_raw;

    if (_beta.is_dependent())
      _J.d(_beta) = Z * N0 * exp_raw;

    if (_dg.is_dependent())
      _J.d(_dg) = -prefactor * exp_raw / (k * T);

    if (const auto * const T_param = nl_param("T"))
      _J.d(*T_param) = prefactor * exp_raw * dg / (k * T * T);

    if (const auto * const N0_param = nl_param("N0"))
      _J.d(*N0_param) = Z * beta * exp_raw;
  }
}
} // namespace neml2
