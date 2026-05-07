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

#include "neml2/models/kwn/SFFKGPrecipitationGrowthRate.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(SFFKGPrecipitationGrowthRate);

OptionSet
SFFKGPrecipitationGrowthRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the SFFK precipitate growth rate.";

  options.add_parameter<Scalar>("radius", "Precipitate radius per size bin");

  options.add_input("projected_diffusivity_sum", "Projected diffusivity sum");

  options.add_parameter<Scalar>("gibbs_free_energy_difference", "Gibbs free energy difference");

  options.add_input("temperature", "Temperature");

  options.add_parameter<Scalar>("gas_constant", "Gas constant");

  options.add_output("growth_rate", "Precipitate growth rate per size bin");

  return options;
}

SFFKGPrecipitationGrowthRate::SFFKGPrecipitationGrowthRate(const OptionSet & options)
  : Model(options),
    _R(declare_parameter<Scalar>("R", "radius", true)),
    _proj_sum(declare_input_variable<Scalar>("projected_diffusivity_sum")),
    _dg(declare_parameter<Scalar>("dg", "gibbs_free_energy_difference", true)),
    _T(declare_input_variable<Scalar>("temperature")),
    _R_g(declare_parameter<Scalar>("R_g", "gas_constant", true)),
    _R_dot(declare_output_variable<Scalar>("growth_rate"))
{
}

void
SFFKGPrecipitationGrowthRate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto nbin = _R.intmd_size(-1);
  const auto R = _R;
  const auto proj_sum = _proj_sum();
  const auto dg = _dg;
  const auto T = _T();
  const auto Rg = _R_g;

  const auto denom = R * Rg * T * proj_sum;

  if (out)
    _R_dot = dg / denom;

  if (dout_din)
  {
    const auto inv_denom = 1.0 / denom;
    const auto rate = dg * inv_denom;

    if (const auto * const R_param = nl_param("R"))
    {
      const auto d_rate_dR = -rate / R;
      const auto r_map = imap_v<Scalar>(_R.options()).intmd_expand(nbin);
      const auto diag_r = intmd_diagonalize(r_map);
      _R_dot.d(*R_param, 2, 1, 1) = d_rate_dR.intmd_unsqueeze(1) * diag_r;
    }

    _R_dot.d(_proj_sum, 1, 1, 0) = -rate / proj_sum;

    if (const auto * const dg_param = nl_param("dg"))
      _R_dot.d(*dg_param, 1, 1, 0) = inv_denom;

    _R_dot.d(_T, 1, 1, 0) = -rate / T;

    if (const auto * const Rg_param = nl_param("R_g"))
      _R_dot.d(*Rg_param, 1, 1, 0) = -rate / Rg;
  }
}
} // namespace neml2
