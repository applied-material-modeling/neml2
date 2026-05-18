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

#include "neml2/models/solid_mechanics/traction_separation_law/OrthotropicLinearTraction.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
register_NEML2_object(OrthotropicLinearTraction);

OptionSet
OrthotropicLinearTraction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Orthotropic linear-elastic interface traction: \\f$ T_n = K_n \\delta_n^\\text{sep} \\f$, "
      "\\f$ T_{si} = K_t \\delta_{si} \\f$. If `normal_penetration` is supplied, "
      "\\f$ K_\\text{pen} \\delta_n^\\text{pen} \\f$ is added to \\f$ T_n \\f$ as a penalty "
      "term resisting interpenetration (`penalty_stiffness` becomes required); otherwise "
      "interpenetration produces zero normal traction.";

  options.add_input("normal_separation",
                    "Normal separation \\f$ \\delta_n^\\text{sep} \\f$ (typically the "
                    "Macaulay-positive part of the normal jump)");
  options.add_input("normal_penetration",
                    "Optional normal penetration \\f$ \\delta_n^\\text{pen} \\f$ (typically the "
                    "Macaulay-negative part of the normal jump). When supplied, "
                    "\\f$ K_\\text{pen} \\delta_n^\\text{pen} \\f$ is added to \\f$ T_n \\f$ "
                    "as a penalty term resisting interpenetration. Requires "
                    "`penalty_stiffness` to be supplied as well.");
  options.add_input("tangential_separation_1",
                    "First tangential separation \\f$ \\delta_{s1} \\f$");
  options.add_input("tangential_separation_2",
                    "Second tangential separation \\f$ \\delta_{s2} \\f$");
  options.add_output("traction", "Traction Vec");
  options.add_parameter<Scalar>("normal_stiffness", "Normal stiffness K_n");
  options.add_parameter<Scalar>("tangential_stiffness", "Tangential stiffness K_t (isotropic)");
  // Optional with default 0 — the constructor asserts this is supplied iff `normal_penetration` is.
  options.add_parameter<Scalar>(
      "penalty_stiffness",
      TensorName<Scalar>("0"),
      "Penalty stiffness used to resist interpenetration. Required when `normal_penetration` "
      "is supplied; ignored otherwise.");

  return options;
}

OrthotropicLinearTraction::OrthotropicLinearTraction(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<Vec>("traction")),
    _dn_sep(declare_input_variable<Scalar>("normal_separation")),
    _dn_pen(options.user_specified("normal_penetration")
                ? &declare_input_variable<Scalar>("normal_penetration")
                : nullptr),
    _ds1(declare_input_variable<Scalar>("tangential_separation_1")),
    _ds2(declare_input_variable<Scalar>("tangential_separation_2")),
    _Kn(declare_parameter<Scalar>("Kn", "normal_stiffness", false)),
    _Kt(declare_parameter<Scalar>("Kt", "tangential_stiffness", false)),
    _Kpen(_dn_pen ? &declare_parameter<Scalar>("Kpen", "penalty_stiffness", false) : nullptr)
{
  neml_assert(options.user_specified("normal_penetration") ==
                  options.user_specified("penalty_stiffness"),
              "OrthotropicLinearTraction: `normal_penetration` and `penalty_stiffness` must be "
              "supplied together (or neither).");
}

void
OrthotropicLinearTraction::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    auto T_n = _Kn * _dn_sep();
    if (_dn_pen)
      T_n = T_n + (*_Kpen) * (*_dn_pen)();
    _to = Vec::fill(T_n, _Kt * _ds1(), _Kt * _ds2());
  }

  if (dout_din)
  {
    const auto zero = Scalar::zeros_like(_dn_sep());
    _to.d(_dn_sep) = Vec::fill(_Kn, zero, zero);
    if (_dn_pen)
      _to.d(*_dn_pen) = Vec::fill(Scalar(*_Kpen), zero, zero);
    _to.d(_ds1) = Vec::fill(zero, Scalar(_Kt), zero);
    _to.d(_ds2) = Vec::fill(zero, zero, Scalar(_Kt));
  }
}
} // namespace neml2
