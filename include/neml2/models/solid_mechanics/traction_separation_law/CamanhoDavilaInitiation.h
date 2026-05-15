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
 * @brief Camanho-Davila mixed-mode initiation displacement jump.
 *
 * In the opening branch (\f$ \delta_n^+ > 0 \f$):
 * \f[
 *   \delta_\text{init} =
 *     \frac{\delta_{n0} \delta_{s0} \sqrt{1 + \beta^2}}
 *          {\sqrt{\delta_{s0}^2 + \beta^2 \delta_{n0}^2}},
 * \f]
 * with \f$ \delta_{n0} = N/K \f$ and \f$ \delta_{s0} = S/K \f$.
 * In the compression branch, \f$ \delta_\text{init} = \delta_{s0} \f$.
 *
 * The mask is taken from `normal > 0` and is detached so the TorchScript
 * tracer does not capture a grad-tracking mask into the JIT graph. The
 * dtype-aware regularizer for the inner `sqrt` comes from
 * `neml2::machine_precision()`.
 */
class CamanhoDavilaInitiation : public Model
{
public:
  static OptionSet expected_options();

  CamanhoDavilaInitiation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Initiation displacement jump \f$ \delta_\text{init} \f$
  Variable<Scalar> & _to;

  /// Mode-mixity ratio \f$ \beta \f$
  const Variable<Scalar> & _beta;

  /// Macaulay-positive normal jump (only needed to determine the opening/compression branch)
  const Variable<Scalar> & _dn_pos;

  /// Penalty stiffness K
  const Scalar & _K;

  /// Tensile (normal) strength N
  const Scalar & _N;

  /// Shear strength S
  const Scalar & _S;
};
} // namespace neml2
