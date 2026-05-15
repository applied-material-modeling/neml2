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

#include "neml2/models/solid_mechanics/traction_separation_law/ExponentialTraction.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/where.h"

namespace neml2
{
register_NEML2_object(ExponentialTraction);

OptionSet
ExponentialTraction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Exponential cohesive-zone traction with internal damage state. Computes "
      "\\f$ d_\\text{trial} = 1 - \\exp(-\\delta_\\text{eff}/\\delta_0) \\f$ from the current "
      "effective separation, caps it for irreversibility against the previous-step value "
      "(auto-declared via `history_name`), and assembles \\f$ T_n = (1-d)(G_c/\\delta_0^2) "
      "\\delta_n^\\text{sep}, T_{si} = (1-d)(G_c/\\delta_0^2) \\delta_{si} \\f$. If "
      "`normal_penetration` is supplied, \\f$ K_\\text{pen}\\,\\delta_n^\\text{pen} \\f$ is "
      "added to \\f$ T_n \\f$ as a penalty term resisting interpenetration "
      "(`penalty_stiffness` becomes required).";

  options.add_input("effective_separation",
                    "Current effective separation \\f$ \\delta_\\text{eff} \\f$ (typically "
                    "computed by an upstream `ScalarPNorm`)");
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
  options.add_output("to", "Traction Vec");
  options.add_output("damage", "Damage scalar (current step, irreversibility-capped)");
  options.add_parameter<Scalar>("fracture_toughness", "Fracture toughness G_c");
  options.add_parameter<Scalar>("characteristic_length",
                                "Softening length scale \\f$ \\delta_0 \\f$");
  // Optional with default 0 — the constructor asserts this is supplied iff `normal_penetration` is.
  options.add_parameter<Scalar>(
      "penalty_stiffness",
      TensorName<Scalar>("0"),
      "Penalty stiffness used to resist interpenetration. Required when `normal_penetration` "
      "is supplied; ignored otherwise.");

  return options;
}

ExponentialTraction::ExponentialTraction(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Vec>("to")),
    _d(declare_output_variable<Scalar>("damage")),
    _d_old(declare_input_variable<Scalar>(history_name(_d.name(), /*nstep=*/1))),
    _delta_eff(declare_input_variable<Scalar>("effective_separation")),
    _dn_sep(declare_input_variable<Scalar>("normal_separation")),
    _dn_pen(options.user_specified("normal_penetration")
                ? &declare_input_variable<Scalar>("normal_penetration")
                : nullptr),
    _ds1(declare_input_variable<Scalar>("tangential_separation_1")),
    _ds2(declare_input_variable<Scalar>("tangential_separation_2")),
    _Gc(declare_parameter<Scalar>("Gc", "fracture_toughness", false)),
    _delta0(declare_parameter<Scalar>("delta0", "characteristic_length", false)),
    _Kpen(_dn_pen ? &declare_parameter<Scalar>("Kpen", "penalty_stiffness", false) : nullptr)
{
  neml_assert(options.user_specified("normal_penetration") ==
                  options.user_specified("penalty_stiffness"),
              "ExponentialTraction: `normal_penetration` and `penalty_stiffness` must be "
              "supplied together (or neither).");
}

void
ExponentialTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // -------- Trial damage from the current effective separation.
  const auto exp_term = neml2::exp(-_delta_eff() / _delta0);
  const auto d_trial = 1.0 - exp_term;

  // -------- Irreversibility cap: damage = max(d_trial, d_old).
  const auto advance = (d_trial > _d_old()).detach();
  const auto d = neml2::where(advance, d_trial, _d_old());

  const auto c = _Gc / (_delta0 * _delta0);
  const auto active_scale = (1.0 - d) * c;

  if (out)
  {
    auto T_n = active_scale * _dn_sep();
    if (_dn_pen)
      T_n = T_n + (*_Kpen) * (*_dn_pen)();
    _to = Vec::fill(T_n, active_scale * _ds1(), active_scale * _ds2());
    _d = d;
  }

  if (!dout_din)
    return;

  // -------- Jacobians.
  // d(d_trial)/d(delta_eff) = (1/delta_0) * exp(-delta_eff/delta_0)
  const auto dt_ddeff = exp_term / _delta0;
  // Irreversibility freezes the partial when d_trial doesn't advance.
  // (Keep `zero`/`one` as Scalar tensors because `where` requires both branches to be
  // typed Tensors of matching shape — literal doubles aren't accepted directly.)
  const auto zero = Scalar::zeros_like(_delta_eff());
  const auto one = Scalar::ones_like(_delta_eff());
  const auto dd_ddeff = neml2::where(advance, dt_ddeff, zero);
  const auto dd_dd_old = neml2::where(advance, zero, one);

  _d.d(_delta_eff) = dd_ddeff;
  _d.d(_d_old) = dd_dd_old;

  // d(traction)/...
  // For each component i, T_active_i = (1-d)*c*δ_i, so
  //   dT_i/d(d)   = -c * δ_i
  //   dT_i/d(δ_i) =  (1-d)*c   (direct — d held fixed)
  const auto dT_dd_n = -c * _dn_sep();
  const auto dT_dd_s1 = -c * _ds1();
  const auto dT_dd_s2 = -c * _ds2();

  _to.d(_delta_eff) = Vec::fill(dT_dd_n * dd_ddeff, dT_dd_s1 * dd_ddeff, dT_dd_s2 * dd_ddeff);
  _to.d(_d_old) = Vec::fill(dT_dd_n * dd_dd_old, dT_dd_s1 * dd_dd_old, dT_dd_s2 * dd_dd_old);

  _to.d(_dn_sep) = Vec::fill(active_scale, zero, zero);
  _to.d(_ds1) = Vec::fill(zero, active_scale, zero);
  _to.d(_ds2) = Vec::fill(zero, zero, active_scale);

  if (_dn_pen)
    _to.d(*_dn_pen) = Vec::fill(Scalar(*_Kpen), zero, zero);
}
} // namespace neml2
