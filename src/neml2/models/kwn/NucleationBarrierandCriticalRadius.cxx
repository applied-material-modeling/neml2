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

#include "neml2/models/kwn/NucleationBarrierandCriticalRadius.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/misc/types.h"

namespace neml2
{
register_NEML2_object(NucleationBarrierandCriticalRadius);

OptionSet
NucleationBarrierandCriticalRadius::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the nucleation critical radius and Gibbs free energy barrier.";

  options.add_parameter<Scalar>("surface_energy", "Surface energy of the precipitate");

  options.add_parameter<Scalar>("total_gibbs_free_energy_difference",
                                "Total Gibbs free energy difference driving nucleation");

  options.add_parameter<Scalar>("molar_volume", "Molar volume of the precipitate");

  options.add_output("nucleation_barrier", "Gibbs free energy barrier for nucleation");

  options.add_output("critical_radius", "Critical radius for nucleation");

  return options;
}

NucleationBarrierandCriticalRadius::NucleationBarrierandCriticalRadius(const OptionSet & options)
  : Model(options),
    _gamma(declare_parameter<Scalar>("gamma", "surface_energy", true)),
    _dg_total(declare_parameter<Scalar>(
        "dg_total", "total_gibbs_free_energy_difference", /*allow_nonlinear=*/true)),
    _V_m(declare_parameter<Scalar>("V_m", "molar_volume")),
    _dg(declare_output_variable<Scalar>("nucleation_barrier")),
    _R_crit(declare_output_variable<Scalar>("critical_radius"))
{
}

void
NucleationBarrierandCriticalRadius::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto gamma = _gamma;
  const auto dg_total = _dg_total;
  const auto V_m = _V_m;

  const auto R_crit = 2.0 * gamma * V_m / dg_total;

  const auto dg =
      (16.0 / 3.0) * neml2::pi * pow(gamma, 3.0) * pow(V_m, 2.0) / (dg_total * dg_total);

  if (out)
  {
    _dg = dg;
    _R_crit = R_crit;
  }

  if (dout_din)
  {
    if (const auto * const gamma_param = nl_param("gamma"))
      _dg.d(*gamma_param) = 3.0 * dg / gamma;

    if (const auto * const dg_param = nl_param("dg_total"))
      _dg.d(*dg_param) = -2.0 * dg / dg_total;

    if (const auto * const gamma_param = nl_param("gamma"))
      _R_crit.d(*gamma_param) = 2.0 * V_m / dg_total;

    if (const auto * const dg_param = nl_param("dg_total"))
      _R_crit.d(*dg_param) = -R_crit / dg_total;
  }
}
} // namespace neml2
