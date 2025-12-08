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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipStrengthMap.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
register_NEML2_object(SingleSlipStrengthMap);

OptionSet
SingleSlipStrengthMap::expected_options()
{
  OptionSet options = SlipStrengthMap::expected_options();

  options.doc() =
      "Calculates the slip system strength for all slip systems as \\f$ \\hat{\\tau}_i = "
      "\\bar{\\tau} + \\tau_0 \\f$ where \\f$ \\hat{\\tau}_i \\f$ is the strength for slip system "
      "i, \\f$ \\bar{\\tau} \\f$ is an evolving slip system strength (one value of all systems), "
      "defined by another object, and \\f$ \\tau_0 \\f$ is a constant strength.";

  options.set_input("slip_hardening") = VariableName(STATE, "internal", "slip_hardening");
  options.set("slip_hardening").doc() = "The name of the evolving, scalar strength";

  options.set_parameter<TensorName<Scalar>>("constant_strength");
  options.set("constant_strength").doc() = "The constant slip system strength";

  return options;
}

SingleSlipStrengthMap::SingleSlipStrengthMap(const OptionSet & options)
  : SlipStrengthMap(options),
    _tau_bar(declare_input_variable<Scalar>("slip_hardening")),
    _tau_const(declare_parameter<Scalar>("constant_strength", "constant_strength", true))
{
}

void
SingleSlipStrengthMap::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
    _tau = (_tau_bar + _tau_const).intmd_unsqueeze(-1);

  if (dout_din)
  {
    if (_tau_bar.is_dependent())
      _tau.d(_tau_bar) = Scalar::ones({}, _tau.intmd_sizes(), _tau_bar.options());

    if (const auto * const tau_const = nl_param("constant_strength"))
      _tau.d(*tau_const) = Scalar::ones({}, _tau.intmd_sizes(), _tau_const.options());
  }
}
} // namespace neml2
