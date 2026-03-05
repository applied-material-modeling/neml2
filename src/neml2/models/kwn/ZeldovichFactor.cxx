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

#include "neml2/models/kwn/ZeldovichFactor.h"

#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
register_NEML2_object(ZeldovichFactor);

OptionSet
ZeldovichFactor::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the Zeldovich factor for nucleation.";

  options.set_input("critical_radius");
  options.set("critical_radius").doc() = "Critical radius for nucleation";

  options.set_parameter<TensorName<Scalar>>("surface_energy");
  options.set("surface_energy").doc() = "Surface energy of the precipitate";

  options.set_parameter<TensorName<Scalar>>("temperature");
  options.set("temperature").doc() = "Temperature";

  options.set_parameter<TensorName<Scalar>>("molar_volume");
  options.set("molar_volume").doc() = "Molar volume of the precipitate";

  options.set_parameter<TensorName<Scalar>>("avogadro_number");
  options.set("avogadro_number").doc() = "Avogadro's number";

  options.set_parameter<TensorName<Scalar>>("boltzmann_constant");
  options.set("boltzmann_constant").doc() = "Boltzmann constant";

  options.set_output("zeldovich_factor");
  options.set("zeldovich_factor").doc() = "Zeldovich factor for nucleation";

  return options;
}

ZeldovichFactor::ZeldovichFactor(const OptionSet & options)
  : Model(options),
    _R_crit(declare_input_variable<Scalar>("critical_radius")),
    _gamma(declare_parameter<Scalar>("gamma", "surface_energy", true)),
    _T(declare_parameter<Scalar>("T", "temperature", /*allow_nonlinear=*/true)),
    _V_m(declare_parameter<Scalar>("V_m", "molar_volume")),
    _N_a(declare_parameter<Scalar>("N_a", "avogadro_number")),
    _k(declare_parameter<Scalar>("k", "boltzmann_constant")),
    _Z(declare_output_variable<Scalar>("zeldovich_factor"))
{
}

void
ZeldovichFactor::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto R_crit = _R_crit();
  const auto gamma = _gamma;
  const auto T = _T;
  const auto V_m = _V_m;
  const auto N_a = _N_a;
  const auto k = _k;

  const auto coef = V_m / (2.0 * neml2::pi * N_a * R_crit * R_crit);
  const auto root = sqrt(gamma / (k * T));
  const auto Z = coef * root;

  if (out)
    _Z = Z;

  if (dout_din)
  {
    if (_R_crit.is_dependent())
      _Z.d(_R_crit) = -2.0 * Z / R_crit;

    if (const auto * const gamma_param = nl_param("gamma"))
      _Z.d(*gamma_param) = 0.5 * Z / gamma;

    if (const auto * const T_param = nl_param("T"))
      _Z.d(*T_param) = -0.5 * Z / T;
  }
}
} // namespace neml2
