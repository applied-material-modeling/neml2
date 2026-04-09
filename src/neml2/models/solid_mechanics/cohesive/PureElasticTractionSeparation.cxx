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

#include "neml2/models/solid_mechanics/cohesive/PureElasticTractionSeparation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"

namespace neml2
{
register_NEML2_object(PureElasticTractionSeparation);

OptionSet
PureElasticTractionSeparation::expected_options()
{
  OptionSet options = TractionSeparationModel::expected_options();
  options.doc() +=
      " following a linear elastic relationship, \\f$ T_n = K_n \\delta_n \\f$ and "
      "\\f$ T_s = K_t \\delta_s \\f$, where \\f$ K_n \\f$ and \\f$ K_t \\f$ are the normal "
      "and tangential stiffnesses, respectively.";

  options.set_parameter<TensorName<Scalar>>("normal_stiffness");
  options.set("normal_stiffness").doc() =
      "Penalty stiffness \\f$ K_n \\f$ relating normal traction to normal opening displacement, "
      "\\f$ T_n = K_n \\delta_n \\f$. Units: stress per length (e.g. MPa/mm).";

  options.set_parameter<TensorName<Scalar>>("tangent_stiffness");
  options.set("tangent_stiffness").doc() =
      "Penalty stiffness \\f$ K_t \\f$ relating tangential traction to sliding displacement "
      "(applied equally in both tangential directions), "
      "\\f$ T_{s1} = K_t \\delta_{s1} \\f$, \\f$ T_{s2} = K_t \\delta_{s2} \\f$. "
      "Units: stress per length (e.g. MPa/mm).";

  return options;
}

PureElasticTractionSeparation::PureElasticTractionSeparation(const OptionSet & options)
  : TractionSeparationModel(options),
    _Kn(declare_parameter<Scalar>("Kn", "normal_stiffness")),
    _Kt(declare_parameter<Scalar>("Kt", "tangent_stiffness"))
{
}

void
PureElasticTractionSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto delta = _delta();
  const auto dn = delta(0);
  const auto ds1 = delta(1);
  const auto ds2 = delta(2);

  if (out)
    _traction = Vec::fill(_Kn * dn, _Kt * ds1, _Kt * ds2);

  if (dout_din)
    if (_delta.is_dependent())
      _traction.d(_delta) = R2::fill(_Kn, _Kt, _Kt);
}
} // namespace neml2
