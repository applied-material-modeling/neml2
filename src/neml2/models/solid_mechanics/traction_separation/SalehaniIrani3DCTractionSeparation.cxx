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

#include "neml2/models/solid_mechanics/traction_separation/SalehaniIrani3DCTractionSeparation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/exp.h"

#include <cmath>

namespace neml2
{
register_NEML2_object(SalehaniIrani3DCTractionSeparation);

OptionSet
SalehaniIrani3DCTractionSeparation::expected_options()
{
  OptionSet options = TractionSeparation::expected_options();
  options.doc() =
      "3D exponential cohesive-zone traction-separation law of Salehani and Irani. "
      "The normal direction enters linearly (\\f$\\alpha=1\\f$) and the two tangential directions "
      "enter quadratically (\\f$\\alpha=2\\f$) in the exponent. Following the reference "
      "implementation, the internal tangential characteristic length is \\f$ \\sqrt{2} \\f$ times "
      "the user-supplied value.";

  options.add_parameter<Scalar>("normal_characteristic_length",
                                "Normal characteristic length (raw user input)");
  options.add_parameter<Scalar>(
      "tangential_characteristic_length",
      "Tangential characteristic length (raw user input; the internal value is sqrt(2) times this)");
  options.add_parameter<Scalar>("maximum_normal_traction", "Maximum normal traction");
  options.add_parameter<Scalar>("maximum_shear_traction", "Maximum shear traction");

  return options;
}

SalehaniIrani3DCTractionSeparation::SalehaniIrani3DCTractionSeparation(const OptionSet & options)
  : TractionSeparation(options),
    _delta_u0_n(declare_parameter<Scalar>("delta_u0_n", "normal_characteristic_length", true)),
    _delta_u0_t(declare_parameter<Scalar>("delta_u0_t", "tangential_characteristic_length", true)),
    _Tmax_n(declare_parameter<Scalar>("Tmax_n", "maximum_normal_traction", true)),
    _Tmax_t(declare_parameter<Scalar>("Tmax_t", "maximum_shear_traction", true))
{
}

void
SalehaniIrani3DCTractionSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  using std::exp;
  using std::sqrt;

  // Internal characteristic-length vector: tangential is sqrt(2) * raw input
  const auto sqrt2 = std::sqrt(2.0);
  const auto delta_u0_t_int = sqrt2 * _delta_u0_t;

  // Prefactors a_i (alpha=1 for normal, alpha=2 for shear)
  const auto e = std::exp(1.0);
  const auto a_n = e * _Tmax_n;
  const auto a_t = std::sqrt(2.0 * e) * _Tmax_t;

  // Normalized jump components b_i = delta_i / delta_u0_i
  const auto delta_n = _delta()(0);
  const auto delta_s1 = _delta()(1);
  const auto delta_s2 = _delta()(2);

  const auto b_n = delta_n / _delta_u0_n;
  const auto b_s1 = delta_s1 / delta_u0_t_int;
  const auto b_s2 = delta_s2 / delta_u0_t_int;

  // Exponent: alpha=1 for normal, alpha=2 for shear
  const auto x = b_n + b_s1 * b_s1 + b_s2 * b_s2;
  const auto exp_x = neml2::exp(-x);

  if (out)
    _T = Vec::fill(a_n * b_n * exp_x, a_t * b_s1 * exp_x, a_t * b_s2 * exp_x);

  if (dout_din)
  {
    // dx/dδ_j: 1/δ0_n for j=n, 2 δ_j/δ0_t^2 for j=s1,s2
    const auto dx_dn = Scalar::ones_like(_delta_u0_n) / _delta_u0_n;
    const auto dx_ds1 = 2.0 * delta_s1 / (delta_u0_t_int * delta_u0_t_int);
    const auto dx_ds2 = 2.0 * delta_s2 / (delta_u0_t_int * delta_u0_t_int);

    // db_i/dδ_j = δ_ij / δ0_i (Kronecker)
    const auto db_n_dn = Scalar::ones_like(_delta_u0_n) / _delta_u0_n;
    const auto db_t_dt = Scalar::ones_like(_delta_u0_n) / delta_u0_t_int;

    // dT_i/dδ_j = a_i * exp_x * (db_i/dδ_j - b_i * dx/dδ_j)
    // Rows: T_n, T_s1, T_s2  Cols: δ_n, δ_s1, δ_s2
    const auto zero = Scalar::zeros_like(_delta_u0_n);

    const auto dTn_dn = a_n * exp_x * (db_n_dn - b_n * dx_dn);
    const auto dTn_ds1 = a_n * exp_x * (zero - b_n * dx_ds1);
    const auto dTn_ds2 = a_n * exp_x * (zero - b_n * dx_ds2);

    const auto dTs1_dn = a_t * exp_x * (zero - b_s1 * dx_dn);
    const auto dTs1_ds1 = a_t * exp_x * (db_t_dt - b_s1 * dx_ds1);
    const auto dTs1_ds2 = a_t * exp_x * (zero - b_s1 * dx_ds2);

    const auto dTs2_dn = a_t * exp_x * (zero - b_s2 * dx_dn);
    const auto dTs2_ds1 = a_t * exp_x * (zero - b_s2 * dx_ds1);
    const auto dTs2_ds2 = a_t * exp_x * (db_t_dt - b_s2 * dx_ds2);

    _T.d(_delta) = R2::fill(dTn_dn,
                            dTn_ds1,
                            dTn_ds2,
                            dTs1_dn,
                            dTs1_ds1,
                            dTs1_ds2,
                            dTs2_dn,
                            dTs2_ds1,
                            dTs2_ds2);
  }
}
} // namespace neml2
