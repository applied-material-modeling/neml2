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

#include "neml2/models/solid_mechanics/traction_separation/PureElasticTractionSeparation.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"

namespace neml2
{
register_NEML2_object(PureElasticTractionSeparation);

OptionSet
PureElasticTractionSeparation::expected_options()
{
  OptionSet options = TractionSeparation::expected_options();
  options.doc() =
      "Purely linear-elastic cohesive-zone traction-separation law: "
      "\\f$ \\boldsymbol{T} = \\operatorname{diag}(K_n, K_t, K_t)\\, \\boldsymbol{\\delta} \\f$. "
      "Treats opening and closing symmetrically; pair with a contact penalty if compression must "
      "be handled separately.";

  options.add_parameter<Scalar>("normal_stiffness", "Elastic stiffness in the normal direction");
  options.add_parameter<Scalar>("tangential_stiffness",
                                "Elastic stiffness in both tangential directions");

  return options;
}

PureElasticTractionSeparation::PureElasticTractionSeparation(const OptionSet & options)
  : TractionSeparation(options),
    _Kn(declare_parameter<Scalar>("Kn", "normal_stiffness", true)),
    _Kt(declare_parameter<Scalar>("Kt", "tangential_stiffness", true))
{
}

void
PureElasticTractionSeparation::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto K = R2::fill(_Kn, _Kt, _Kt);

  if (out)
    _T = K * _delta;

  if (dout_din)
  {
    _T.d(_delta) = K;

    const auto zero = Scalar::zeros_like(_Kn);
    const auto one = Scalar::ones_like(_Kn);

    if (const auto * const Kn = nl_param("Kn"))
      _T.d(*Kn) = R2::fill(one, zero, zero) * _delta();

    if (const auto * const Kt = nl_param("Kt"))
      _T.d(*Kt) = R2::fill(zero, one, one) * _delta();
  }
}
} // namespace neml2
