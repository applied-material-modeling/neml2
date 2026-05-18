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

#include "neml2/models/solid_mechanics/traction_separation_law/SalehaniIraniTraction.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/where.h"

#include <cmath>

namespace neml2
{
register_NEML2_object(SalehaniIraniTraction);

OptionSet
SalehaniIraniTraction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "3D coupled exponential cohesive law of Salehani & Irani with internal damage state. "
      "Computes \\f$ d_\\text{trial} = 1 - \\exp(-x) \\f$ where \\f$ x = b_n + b_{s1}^2 + "
      "b_{s2}^2 \\f$, caps it for irreversibility against the previous-step value "
      "(auto-declared via `history_name`), and assembles \\f$ T_i = a_i b_i (1 - d) \\f$. "
      "For monotonic loading this is exactly equivalent to the original \\f$ T_i = a_i b_i "
      "\\exp(-x) \\f$; under unloading the damage cap freezes the softness at its historical "
      "peak. The internal tangential characteristic length is \\f$ \\sqrt{2}\\,\\delta_{u0,t} "
      "\\f$. If `normal_penetration` is supplied, \\f$ K_\\text{pen}\\,\\delta_n^\\text{pen} "
      "\\f$ is added to \\f$ T_n \\f$ (`penalty_stiffness` becomes required).";

  options.add_input("normal_separation",
                    "Normal separation \\f$ \\delta_n^\\text{sep} \\f$ (typically the "
                    "Macaulay-positive part of the normal jump)");
  options.add_input("normal_penetration",
                    "Optional normal penetration \\f$ \\delta_n^\\text{pen} \\f$. When "
                    "supplied, \\f$ K_\\text{pen}\\,\\delta_n^\\text{pen} \\f$ is added to "
                    "\\f$ T_n \\f$ as a penalty term resisting interpenetration. Requires "
                    "`penalty_stiffness` to be supplied as well.");
  options.add_input("tangential_separation_1",
                    "First tangential separation \\f$ \\delta_{s1} \\f$");
  options.add_input("tangential_separation_2",
                    "Second tangential separation \\f$ \\delta_{s2} \\f$");
  options.add_output("traction", "Traction Vec");
  options.add_output("damage", "Damage scalar (current step, irreversibility-capped)");
  options.add_parameter<Scalar>("normal_characteristic_length",
                                "Normal characteristic length (raw user input)");
  options.add_parameter<Scalar>("tangential_characteristic_length",
                                "Tangential characteristic length (raw user input; the internal "
                                "value is sqrt(2) times this)");
  options.add_parameter<Scalar>("normal_strength", "Normal strength (peak normal traction)");
  options.add_parameter<Scalar>("shear_strength", "Shear strength (peak shear traction)");
  // Optional with default 0 — the constructor asserts this is supplied iff `normal_penetration` is.
  options.add_parameter<Scalar>(
      "penalty_stiffness",
      TensorName<Scalar>("0"),
      "Penalty stiffness used to resist interpenetration. Required when `normal_penetration` "
      "is supplied; ignored otherwise.");

  return options;
}

SalehaniIraniTraction::SalehaniIraniTraction(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Vec>("traction")),
    _d(declare_output_variable<Scalar>("damage")),
    _d_old(declare_input_variable<Scalar>(history_name(_d.name(), /*nstep=*/1))),
    _dn_sep(declare_input_variable<Scalar>("normal_separation")),
    _dn_pen(options.user_specified("normal_penetration")
                ? &declare_input_variable<Scalar>("normal_penetration")
                : nullptr),
    _ds1(declare_input_variable<Scalar>("tangential_separation_1")),
    _ds2(declare_input_variable<Scalar>("tangential_separation_2")),
    _delta_u0_n(declare_parameter<Scalar>("delta_u0_n", "normal_characteristic_length", false)),
    _delta_u0_t(declare_parameter<Scalar>("delta_u0_t", "tangential_characteristic_length", false)),
    _Tmax_n(declare_parameter<Scalar>("Tmax_n", "normal_strength", false)),
    _Tmax_t(declare_parameter<Scalar>("Tmax_t", "shear_strength", false)),
    _Kpen(_dn_pen ? &declare_parameter<Scalar>("Kpen", "penalty_stiffness", false) : nullptr)
{
  neml_assert(options.user_specified("normal_penetration") ==
                  options.user_specified("penalty_stiffness"),
              "SalehaniIraniTraction: `normal_penetration` and `penalty_stiffness` must be "
              "supplied together (or neither).");
}

void
SalehaniIraniTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Internal characteristic-length vector: tangential is sqrt(2) * raw input.
  const auto delta_u0_t_int = sqrt2 * _delta_u0_t;

  // Prefactors a_i (normal enters linearly, shear enters quadratically).
  const auto e = std::exp(1.0);
  const auto a_n = e * _Tmax_n;
  const auto a_t = std::sqrt(2.0 * e) * _Tmax_t;

  const auto b_n = _dn_sep() / _delta_u0_n;
  const auto b_s1 = _ds1() / delta_u0_t_int;
  const auto b_s2 = _ds2() / delta_u0_t_int;

  const auto x = b_n + b_s1 * b_s1 + b_s2 * b_s2;
  const auto exp_x = neml2::exp(-x);

  // -------- Trial damage from the exponent.
  const auto d_trial = 1.0 - exp_x;

  // -------- Irreversibility cap.
  const auto advance = (d_trial > _d_old()).detach();
  const auto d = neml2::where(advance, d_trial, _d_old());
  // factor == exp_x in the advancing branch, == (1 - d_old) in the frozen branch.
  const auto factor = 1.0 - d;

  if (out)
  {
    auto T_n = a_n * b_n * factor;
    if (_dn_pen)
      T_n = T_n + (*_Kpen) * (*_dn_pen)();
    _to = Vec::fill(T_n, a_t * b_s1 * factor, a_t * b_s2 * factor);
    _d = d;
  }

  if (!dout_din)
    return;

  // -------- Jacobians.
  // db_i/dδ_j = δ_ij / δ0_i (Kronecker), built as Scalar via 1/<param>.
  // dx/dδ_j: 1/δ0_n for j=n, 2 δ_j/δ0_t² for j=s1,s2.
  const auto db_n_dn = 1.0 / _delta_u0_n;
  const auto db_t_dt = 1.0 / delta_u0_t_int;
  const auto dx_dn = 1.0 / _delta_u0_n;
  const auto dx_ds1 = 2.0 * _ds1() / (delta_u0_t_int * delta_u0_t_int);
  const auto dx_ds2 = 2.0 * _ds2() / (delta_u0_t_int * delta_u0_t_int);

  // Scalar `zero`/`one` are needed for the `where` branches and for `Vec::fill` when mixed
  // with other Scalar arguments (literal doubles can't be substituted there — `where` requires
  // both branches to be Tensors of matching type, and `Vec::fill` rejects mixed Tensor+literal).
  const auto zero = Scalar::zeros_like(_dn_sep());
  const auto one = Scalar::ones_like(_dn_sep());

  // d(damage)/d(input) = where(advance, d(d_trial)/d(input), 0)
  //   d(d_trial)/dδ_j = exp_x * dx/dδ_j
  // d(damage)/d(d_old) = where(advance, 0, 1)
  const auto dd_ddn = neml2::where(advance, exp_x * dx_dn, zero);
  const auto dd_dds1 = neml2::where(advance, exp_x * dx_ds1, zero);
  const auto dd_dds2 = neml2::where(advance, exp_x * dx_ds2, zero);
  const auto dd_dd_old = neml2::where(advance, zero, one);

  _d.d(_dn_sep) = dd_ddn;
  _d.d(_ds1) = dd_dds1;
  _d.d(_ds2) = dd_dds2;
  _d.d(_d_old) = dd_dd_old;

  // T_i = a_i * b_i * factor where factor = (1 - damage). For input δ_j:
  //   dT_i/dδ_j = a_i * (db_i/dδ_j) * factor - a_i * b_i * d(damage)/dδ_j
  // For input d_old (only contributes through factor):
  //   dT_i/dd_old = -a_i * b_i * d(damage)/dd_old
  _to.d(_dn_sep) = Vec::fill(
      a_n * (factor * db_n_dn - b_n * dd_ddn), -a_t * b_s1 * dd_ddn, -a_t * b_s2 * dd_ddn);
  _to.d(_ds1) = Vec::fill(
      -a_n * b_n * dd_dds1, a_t * (factor * db_t_dt - b_s1 * dd_dds1), -a_t * b_s2 * dd_dds1);
  _to.d(_ds2) = Vec::fill(
      -a_n * b_n * dd_dds2, -a_t * b_s1 * dd_dds2, a_t * (factor * db_t_dt - b_s2 * dd_dds2));
  _to.d(_d_old) =
      Vec::fill(-a_n * b_n * dd_dd_old, -a_t * b_s1 * dd_dd_old, -a_t * b_s2 * dd_dd_old);

  if (_dn_pen)
    _to.d(*_dn_pen) = Vec::fill(Scalar(*_Kpen), zero, zero);
}
} // namespace neml2
