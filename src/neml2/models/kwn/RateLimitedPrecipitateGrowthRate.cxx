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

#include "neml2/models/kwn/RateLimitedPrecipitateGrowthRate.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/diagonalize.h"
#include "neml2/tensors/functions/imap.h"

namespace neml2
{
register_NEML2_object(RateLimitedPrecipitateGrowthRate);

OptionSet
RateLimitedPrecipitateGrowthRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Compute the rate-limited precipitate growth rate for a single species.";

  options.set_input("radius");
  options.set("radius").doc() = "Precipitate radius per size bin";

  options.set_input("current_concentration");
  options.set("current_concentration").doc() = "Current concentration in solution";

  options.set_parameter<TensorName<Scalar>>("equilibrium_concentration");
  options.set("equilibrium_concentration").doc() = "Equilibrium concentration in solution";

  options.set_parameter<TensorName<Scalar>>("precipitate_concentration");
  options.set("precipitate_concentration").doc() = "Precipitate concentration";

  options.set_parameter<TensorName<Scalar>>("diffusivity");
  options.set("diffusivity").doc() = "Species diffusivity in solution";

  options.set_output("growth_rate");
  options.set("growth_rate").doc() = "Precipitate growth rate per size bin";

  return options;
}

RateLimitedPrecipitateGrowthRate::RateLimitedPrecipitateGrowthRate(const OptionSet & options)
  : Model(options),
    _R(declare_input_variable<Scalar>("radius")),
    _x(declare_input_variable<Scalar>("current_concentration")),
    _x_eq(declare_parameter<Scalar>("x_eq", "equilibrium_concentration", true)),
    _x_p(declare_parameter<Scalar>("x_p", "precipitate_concentration", true)),
    _D(declare_parameter<Scalar>("D", "diffusivity", true)),
    _R_dot(declare_output_variable<Scalar>("growth_rate"))
{
}

void
RateLimitedPrecipitateGrowthRate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto nbin = _R.intmd_size(-1);
  const auto R = _R();
  const auto x_inf = (_x.intmd_dim() == 0 ? _x().intmd_expand(nbin) : _x());
  const auto x_eq = (_x_eq.intmd_dim() == 0 ? _x_eq.intmd_expand(nbin) : _x_eq);
  const auto x_p = (_x_p.intmd_dim() == 0 ? _x_p.intmd_expand(nbin) : _x_p);
  const auto D = (_D.intmd_dim() == 0 ? _D.intmd_expand(nbin) : _D);

  const auto denom = x_p - x_eq;
  const auto numer = x_inf - x_eq;
  const auto coef = D / R;

  if (out)
    _R_dot = coef * numer / denom;

  if (dout_din)
  {
    const auto denom2 = denom * denom;

    if (_R.is_dependent())
    {
      const auto d_rate_dR = -coef * numer / (denom * R);
      const auto r_map = imap_v<Scalar>(_R.options()).intmd_expand(nbin);
      const auto diag_r = intmd_diagonalize(r_map);
      _R_dot.d(_R, 2, 1, 1) = d_rate_dR.intmd_unsqueeze(1) * diag_r;
    }

    if (_x.is_dependent())
    {
      const auto d_rate_dx = coef / denom;
      if (_x.intmd_dim() == 0)
        _R_dot.d(_x, 1, 1, 0) = d_rate_dx;
      else
      {
        const auto x_map = imap_v<Scalar>(_x.options()).intmd_expand(nbin);
        const auto diag_x = intmd_diagonalize(x_map);
        _R_dot.d(_x, 2, 1, 1) = d_rate_dx.intmd_unsqueeze(1) * diag_x;
      }
    }

    if (const auto * const x_eq_param = nl_param("x_eq"))
    {
      const auto d_rate_dx_eq = D * (x_inf - x_p) / (R * denom2);
      if (x_eq_param->intmd_dim() == 0)
        _R_dot.d(*x_eq_param, 1, 1, 0) = d_rate_dx_eq;
      else
      {
        const auto xeq_map = imap_v<Scalar>(_x_eq.options()).intmd_expand(nbin);
        const auto diag_xeq = intmd_diagonalize(xeq_map);
        _R_dot.d(*x_eq_param, 2, 1, 1) = d_rate_dx_eq.intmd_unsqueeze(1) * diag_xeq;
      }
    }

    if (const auto * const x_p_param = nl_param("x_p"))
    {
      const auto d_rate_dx_p = -D * numer / (R * denom2);
      if (x_p_param->intmd_dim() == 0)
        _R_dot.d(*x_p_param, 1, 1, 0) = d_rate_dx_p;
      else
      {
        const auto xp_map = imap_v<Scalar>(_x_p.options()).intmd_expand(nbin);
        const auto diag_xp = intmd_diagonalize(xp_map);
        _R_dot.d(*x_p_param, 2, 1, 1) = d_rate_dx_p.intmd_unsqueeze(1) * diag_xp;
      }
    }

    if (const auto * const D_param = nl_param("D"))
    {
      const auto d_rate_dD = numer / (R * denom);
      if (D_param->intmd_dim() == 0)
        _R_dot.d(*D_param, 1, 1, 0) = d_rate_dD;
      else
      {
        const auto d_map = imap_v<Scalar>(_D.options()).intmd_expand(nbin);
        const auto diag_d = intmd_diagonalize(d_map);
        _R_dot.d(*D_param, 2, 1, 1) = d_rate_dD.intmd_unsqueeze(1) * diag_d;
      }
    }
  }
}
} // namespace neml2
