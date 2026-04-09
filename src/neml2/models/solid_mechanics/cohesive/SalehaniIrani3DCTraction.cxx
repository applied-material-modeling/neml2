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

#include "neml2/models/solid_mechanics/cohesive/SalehaniIrani3DCTraction.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/sqrt.h"

#include <cmath>

namespace neml2
{
register_NEML2_object(SalehaniIrani3DCTraction);

OptionSet
SalehaniIrani3DCTraction::expected_options()
{
  OptionSet options = TractionSeparationModel::expected_options();
  options.doc() +=
      " following the Salehani-Irani 2018 exponential cohesive zone law. "
      "The traction components are \\f$ T_i = a_i (\\delta_i / \\delta_{i,0}) "
      "\\exp(-x) \\f$ where \\f$ x = \\delta_n/\\delta_{n,0} + (\\delta_{s1}/\\delta_{t,0})^2 "
      "+ (\\delta_{s2}/\\delta_{t,0})^2 \\f$.";

  options.set_parameter<TensorName<Scalar>>("normal_gap_at_maximum_normal_traction");
  options.set("normal_gap_at_maximum_normal_traction").doc() =
      "Characteristic normal gap \\f$ \\delta_{n,0} \\f$ at which normal traction is maximum";

  options.set_parameter<TensorName<Scalar>>("tangential_gap_at_maximum_shear_traction");
  options.set("tangential_gap_at_maximum_shear_traction").doc() =
      "Characteristic tangential gap at which shear traction is maximum; "
      "stored internally as \\f$ \\delta_{t,0} = \\sqrt{2} \\cdot \\delta_{t,\\text{in}} \\f$";

  options.set_parameter<TensorName<Scalar>>("maximum_normal_traction");
  options.set("maximum_normal_traction").doc() = "Maximum normal traction \\f$ T_{n,\\max} \\f$";

  options.set_parameter<TensorName<Scalar>>("maximum_shear_traction");
  options.set("maximum_shear_traction").doc() = "Maximum shear traction \\f$ T_{s,\\max} \\f$";

  return options;
}

SalehaniIrani3DCTraction::SalehaniIrani3DCTraction(const OptionSet & options)
  : TractionSeparationModel(options),
    _delta_n0(declare_parameter<Scalar>("delta_n0", "normal_gap_at_maximum_normal_traction")),
    _delta_t0_raw(
        declare_parameter<Scalar>("delta_t0_raw", "tangential_gap_at_maximum_shear_traction")),
    _T_max_n(declare_parameter<Scalar>("T_max_n", "maximum_normal_traction")),
    _T_max_t(declare_parameter<Scalar>("T_max_t", "maximum_shear_traction"))
{
}

void
SalehaniIrani3DCTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto delta = _delta();
  const auto dn = delta(0);
  const auto ds1 = delta(1);
  const auto ds2 = delta(2);

  // delta_t0 = sqrt(2) * raw input (spec convention)
  const auto sqrt2 = Scalar::full(std::sqrt(2.0), _delta_t0_raw.options());
  const auto delta_t0 = sqrt2 * _delta_t0_raw;

  // Prefactors: a0 = e * T_max_n, a12 = sqrt(2e) * T_max_t
  const auto e = Scalar::full(std::exp(1.0), _T_max_n.options());
  const auto a0 = e * _T_max_n;
  const auto a12 = neml2::sqrt(2 * e) * _T_max_t;

  // Normalised displacements: b_i = delta_i / delta_i0
  const auto b0 = dn / _delta_n0;
  const auto b1 = ds1 / delta_t0;
  const auto b2 = ds2 / delta_t0;

  // Coupling exponent: x = b0 + b1^2 + b2^2
  const auto x = b0 + b1 * b1 + b2 * b2;
  const auto exp_neg_x = neml2::exp(-x);

  if (out)
    _traction = Vec::fill(a0 * b0 * exp_neg_x, a12 * b1 * exp_neg_x, a12 * b2 * exp_neg_x);

  if (dout_din)
    if (_delta.is_dependent())
    {
      // dx/ddelta = (1/delta_n0, 2*ds1/delta_t0^2, 2*ds2/delta_t0^2)
      const auto inv_dn0 = Scalar::full(1.0, dn.options()) / _delta_n0;
      const auto inv_dt0 = Scalar::full(1.0, dn.options()) / delta_t0;
      const auto inv_dt02 = inv_dt0 / delta_t0;

      const auto dx0 = inv_dn0;
      const auto dx1 = 2 * ds1 * inv_dt02;
      const auto dx2 = 2 * ds2 * inv_dt02;

      // dTi/duj = a_i * exp(-x) * (dbi/duj - b_i * dx/duj)
      // Diagonal terms (i == j):
      const auto d00 = a0 * exp_neg_x * (inv_dn0 - b0 * dx0);
      const auto d11 = a12 * exp_neg_x * (inv_dt0 - b1 * dx1);
      const auto d22 = a12 * exp_neg_x * (inv_dt0 - b2 * dx2);
      // Off-diagonal terms:
      const auto d01 = a0 * exp_neg_x * (-b0 * dx1);   // dT_n / dds1
      const auto d02 = a0 * exp_neg_x * (-b0 * dx2);   // dT_n / dds2
      const auto d10 = a12 * exp_neg_x * (-b1 * dx0);  // dT_s1 / ddn
      const auto d12 = a12 * exp_neg_x * (-b1 * dx2);  // dT_s1 / dds2
      const auto d20 = a12 * exp_neg_x * (-b2 * dx0);  // dT_s2 / ddn
      const auto d21 = a12 * exp_neg_x * (-b2 * dx1);  // dT_s2 / dds1

      // Row-major 3x3: (d00, d01, d02, d10, d11, d12, d20, d21, d22)
      _traction.d(_delta) = R2::fill(d00, d01, d02, d10, d11, d12, d20, d21, d22);
    }
}
} // namespace neml2
