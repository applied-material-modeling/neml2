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

#include "neml2/models/solid_mechanics/crystal_plasticity/PowerLawSlipRule.h"
#include "neml2/models/solid_mechanics/crystal_plasticity/SlipRule.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/diag_embed.h"
#include "neml2/tensors/functions/log.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/sign.h"

namespace neml2
{
register_NEML2_object(PowerLawSlipRule);

OptionSet
PowerLawSlipRule::expected_options()
{
  OptionSet options = SlipRule::expected_options();
  options.doc() =
      "Power law slip rule defined as \\f$ \\dot{\\gamma}_i = \\dot{\\gamma}_0 \\left\\langle "
      "\frac{\\left|\\tau_i - \\hat{\\tau}_i^{(kin)}\\right| - "
      "\\hat{\\tau}_i^{(iso)}}{\\hat{\\tau}} "
      "\right\\rangle^n \\operatorname{sign}\\left(\\tau_i - \\hat{\\tau}_i^{(kin)}\\right) \\f$ "
      "with "
      "\\f$ \\dot{\\gamma}_i \\f$ the slip rate on system \\f$ i \\f$, \\f$ \\tau_i \\f$ the "
      "resolved shear, \\f$ \\hat{\\tau}_i \\f$ the slip system strength, \\f$ "
      "\\hat{\\tau}_i^{(kin)} \\f$ is an optional kinematic backstress, \\f$ "
      "\\hat{\\tau}_i^{(iso)} \\f$ is an optional isotropic hardening resistance, \\f$ n \\f$ the "
      "rate "
      "senstivity, and \\f$ \\dot{\\gamma}_0 \\f$ a reference slip rate.";

  options.set_parameter<TensorName<Scalar>>("gamma0");
  options.set("gamma0").doc() = "Reference slip rate";

  options.set_parameter<TensorName<Scalar>>("n");
  options.set("n").doc() = "Rate sensitivity exponent";

  return options;
}

PowerLawSlipRule::PowerLawSlipRule(const OptionSet & options)
  : SlipRule(options),
    _gamma0(declare_parameter<Scalar>("gamma0", "gamma0", true)),
    _n(declare_parameter<Scalar>("n", "n", true))
{
}

void
PowerLawSlipRule::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  const auto D = utils::broadcast_batch_dim(_rss, _tau, _gamma0, _n);

  auto X = _X ? *_X : Scalar::zeros_like(_tau);
  auto R = _R ? *_R : Scalar::zeros_like(_tau);

  if (out)
    _g = _gamma0 * pow(macaulay((abs(_rss - X) - R) / _tau), _n) * sign(_rss - X);

  if (dout_din)
  {
    if (_rss.is_dependent())
      _g.d(_rss) =
          Tensor(batch_diag_embed(_gamma0 * _n *
                                  pow(macaulay((abs(_rss - X) - R) / _tau), _n - 1.0) / _tau),
                 D);

    if (_tau.is_dependent())
      _g.d(_tau) =
          Tensor(batch_diag_embed(-_n * _gamma0 * pow(macaulay((abs(_rss - X) - R) / _tau), _n) /
                                  _tau * sign(_rss - X)),
                 D);

    if (_X && _X->is_dependent())
      _g.d(*_X) =
          Tensor(batch_diag_embed(_gamma0 * _n * pow(macaulay((abs(_rss - X) - R) / _tau), _n) /
                                  (R - abs(_rss - X))),
                 D);

    if (_R && _R->is_dependent())
      _g.d(*_R) = Tensor(batch_diag_embed(-_gamma0 * _n *
                                          pow(macaulay((abs(_rss - X) - R) / _tau), _n - 1) / _tau *
                                          sign(_rss - X)),
                         D);

    if (const auto * const gamma0 = nl_param("gamma0"))
      _g.d(*gamma0) = Tensor(pow(macaulay((abs(_rss - X) - R) / _tau), _n) * sign(_rss - X), D);

    // It should actually be macaulay under the log, not abs.  But this prevents log(0)
    if (const auto * const n = nl_param("n"))
      _g.d(*n) = Tensor(_gamma0 * pow(macaulay((abs(_rss - X) - R) / _tau), _n) * sign(_rss - X) *
                            log(abs((abs(_rss - X) - R) / _tau)),
                        D);
  }
}
} // namespace neml2
