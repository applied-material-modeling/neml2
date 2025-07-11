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

#include "neml2/models/chemical_reactions/ContractingGeometry.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/clamp.h"

namespace neml2
{
register_NEML2_object(ContractingGeometry);
OptionSet
ContractingGeometry::expected_options()
{
  OptionSet options = ReactionMechanism::expected_options();
  options.doc() =
      "The contracting geometry model, often encountered in non-isothermal decomposition or "
      "solid-gas reactions, takes the form of \\f$ f = k(1-a)^n \\f$, where \\f$ k \\f$ is the "
      "reaction coefficient (often temperature-dependent), \\f$ n \\f$ is the reaction order, and "
      "\\f$ a \\f$ is the degree of conversion.";

  options.set_parameter<TensorName<Scalar>>("reaction_coef");
  options.set("reaction_coef").doc() = "Reaction coefficient, k";

  options.set_parameter<TensorName<Scalar>>("reaction_order");
  options.set("reaction_order").doc() = "Reaction order, n";

  return options;
}

ContractingGeometry::ContractingGeometry(const OptionSet & options)
  : ReactionMechanism(options),
    _k(declare_parameter<Scalar>("k", "reaction_coef", /*allow_nonlinear=*/true)),
    _n(declare_parameter<Scalar>("n", "reaction_order"))
{
}

void
ContractingGeometry::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto eps = machine_precision(_a.scalar_type()).toDouble();
  const auto aclamp = clamp(1.0 - _a, 0.0 + eps, 1.0 - eps);

  if (out)
    _f = _k * pow(aclamp, _n);

  if (dout_din)
  {
    if (_a.is_dependent())
      _f.d(_a) = -1.0 * _k * _n * pow(aclamp, _n - 1);

    if (const auto * const k = nl_param("k"))
      _f.d(*k) = pow(aclamp, _n);
  }
}
}
