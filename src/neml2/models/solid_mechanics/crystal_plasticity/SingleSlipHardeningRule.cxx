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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipHardeningRule.h"

#include "neml2/tensors/Scalar.h"

namespace neml2
{
OptionSet
SingleSlipHardeningRule::expected_options()
{
  OptionSet options = Model::expected_options();

  options.doc() =
      "Parent class of slip hardening rules where all slip systems share the same strength.";

  options.set_output("slip_hardening_rate") =
      VariableName(STATE, "internal", "slip_hardening_rate");
  options.set("slip_hardening_rate").doc() =
      "Name of tensor to output the slip system hardening rates into";

  options.set_input("slip_hardening") = VariableName(STATE, "internal", "slip_hardening");
  options.set("slip_hardening").doc() = "Name of current values of slip hardening";

  options.set_input("sum_slip_rates") = VariableName(STATE, "internal", "sum_slip_rates");
  options.set("sum_slip_rates").doc() = "Name of tensor containing the sum of the slip rates";

  return options;
}

SingleSlipHardeningRule::SingleSlipHardeningRule(const OptionSet & options)
  : Model(options),
    _tau_dot(declare_output_variable<Scalar>("slip_hardening_rate")),
    _tau(declare_input_variable<Scalar>("slip_hardening")),
    _gamma_dot_sum(declare_input_variable<Scalar>("sum_slip_rates"))
{
}
} // namespace neml2
