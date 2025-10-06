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

#include "neml2/models/solid_mechanics/crystal_plasticity/VocePerSlipHardeningRule.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/functions/diag_embed.h"

namespace neml2
{
register_NEML2_object(VocePerSlipHardeningRule);

OptionSet
VocePerSlipHardeningRule::expected_options()
{
  OptionSet options = PerSlipHardeningRule::expected_options();
  options.doc() = "Voce hardening type model defined by \\f$ \\dot{\\tau_i} "
                  "= \\theta_0 \\left( 1 - \\frac{\\tau_i}{\\tau_f} \\right) "
                  "\\left| \\dot{\\gamma}_i \\right| \\f$ where \\f$ "
                  "\\theta_0 \\f$ is the initial rate of work hardening, \\f$ \\tau_f \\f$ is the "
                  "saturated, maximum value of the slip system strength, and \\f$ \\dot{\\gamma}_i "
                  "\\f$ is the slip rate on each system.";

  options.set_parameter<TensorName<Scalar>>("initial_slope");
  options.set("initial_slope").doc() = "The initial rate of hardening";
  options.set_parameter<TensorName<Scalar>>("saturated_hardening");
  options.set("saturated_hardening").doc() =
      "The final, saturated value of the slip system strength";
  return options;
}

VocePerSlipHardeningRule::VocePerSlipHardeningRule(const OptionSet & options)
  : PerSlipHardeningRule(options),
    _theta_0(declare_parameter<Scalar>("initial_slope", "initial_slope", true)),
    _tau_f(declare_parameter<Scalar>("saturated_hardening", "saturated_hardening", true))
{
}

void
VocePerSlipHardeningRule::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto D = utils::broadcast_batch_dim(_tau_f, _tau, _gamma_dot, _theta_0);

  if (out)
    _tau_dot = _theta_0 * (1 - _tau / _tau_f) * abs(_gamma_dot);

  if (dout_din)
  {
    if (_tau.is_dependent())
      _tau_dot.d(_tau) = Tensor(batch_diag_embed(-_theta_0 / _tau_f * abs(_gamma_dot)), D);

    if (_gamma_dot.is_dependent())
      _tau_dot.d(_gamma_dot) =
          Tensor(batch_diag_embed(_theta_0 * (1 - _tau / _tau_f) * sign(_gamma_dot)), D);

    if (const auto * const theta_0 = nl_param("initial_slope"))
      _tau_dot.d(*theta_0) = Tensor((1 - _tau / _tau_f) * abs(_gamma_dot), D);

    if (const auto * const tau_f = nl_param("saturated_hardening"))
      _tau_dot.d(*tau_f) = Tensor(_theta_0 * _tau / (_tau_f * _tau_f) * abs(_gamma_dot), D);
  }
}
} // namespace neml2
