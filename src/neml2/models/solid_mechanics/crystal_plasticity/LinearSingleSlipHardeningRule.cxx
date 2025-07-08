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

#include "neml2/models/solid_mechanics/crystal_plasticity/LinearSingleSlipHardeningRule.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(LinearSingleSlipHardeningRule);

OptionSet
LinearSingleSlipHardeningRule::expected_options()
{
  OptionSet options = SingleSlipHardeningRule::expected_options();

  options.doc() = "Simple linear slip system hardening defined by \\f$ \\dot{\\tau} = \\theta "
                  "\\sum_{i=1}^{n_{slip}} \\left| \\dot{\\gamma}_i \\right| \\f$ where \\f$ "
                  "\\theta \\f$ is the hardening slope.";

  options.set_parameter<TensorName<Scalar>>("hardening_slope");
  options.set("hardening_slope").doc() = "Hardening rate";

  return options;
}

LinearSingleSlipHardeningRule::LinearSingleSlipHardeningRule(const OptionSet & options)
  : SingleSlipHardeningRule(options),
    _theta(declare_parameter<Scalar>("hardening_slope", "hardening_slope", true))
{
}

void
LinearSingleSlipHardeningRule::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _tau_dot = _theta * _gamma_dot_sum;

  if (dout_din)
  {
    if (_gamma_dot_sum.is_dependent())
      _tau_dot.d(_gamma_dot_sum) = _theta;

    if (const auto * const theta = nl_param("hardening_slope"))
      _tau_dot.d(*theta) = _gamma_dot_sum;
  }
}
} // namespace neml2
