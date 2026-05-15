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

#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

/**
 * @brief Mode-mixity ratio \f$ \beta = \delta_s / \delta_n^+ \f$ in the
 *        opening branch; \f$ \beta = 0 \f$ in compression.
 *
 * Uses a `where`-and-detach safe-divisor pattern so the masked-off
 * compression branch doesn't trip on a division by zero. The condition is
 * detached so the TorchScript tracer doesn't try to capture a grad-tracking
 * mask into the JIT graph.
 *
 * The `normal` input is expected to be the Macaulay (non-negative) part of
 * the normal jump — typically the `to_positive` output of `MacaulaySplit`.
 */
class ModeMixity : public Model
{
public:
  static OptionSet expected_options();

  ModeMixity(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Mode-mixity ratio \f$ \beta \f$
  Variable<Scalar> & _to;

  /// Macaulay-positive normal jump \f$ \delta_n^+ \f$
  const Variable<Scalar> & _dn_pos;

  /// Tangential magnitude \f$ \delta_s \f$
  const Variable<Scalar> & _ds;
};
} // namespace neml2
