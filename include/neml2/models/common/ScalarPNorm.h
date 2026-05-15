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
 * @brief Weighted \f$ p \f$-norm of an arbitrary number of `Scalar` inputs:
 *        \f$ y = \left( \sum_i w_i |x_i|^p + \varepsilon \right)^{1/p} \f$.
 *
 * The weights \f$ w_i \f$ default to 1 (giving the standard \f$ p \f$-norm)
 * but can be set per-input via the `weights` option, exactly as in
 * `LinearCombination`. With \f$ p = 2 \f$ and unit weights this is the
 * regularized Euclidean norm. With \f$ p = 2 \f$ and non-uniform weights
 * \f$ \boldsymbol{w} = [1, \beta_w, \beta_w] \f$ this captures the
 * Exp-style effective separation
 * \f$ \delta_\text{eff} = \sqrt{\delta_n^2 + \beta_w(\delta_{s1}^2 + \delta_{s2}^2) + \varepsilon}
 * \f$.
 *
 * The dtype-aware regularizer \f$ \varepsilon \f$ comes from
 * `neml2::machine_precision()` and keeps the derivative bounded at the
 * origin. The Jacobian is
 * \f$ \partial y / \partial x_i = w_i \operatorname{sign}(x_i)\, |x_i|^{p-1}\, y^{1-p} \f$,
 * which simplifies to \f$ w_i x_i / y \f$ for \f$ p = 2 \f$.
 */
class ScalarPNorm : public Model
{
public:
  static OptionSet expected_options();

  ScalarPNorm(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The weighted \f$ p \f$-norm output
  Variable<Scalar> & _to;

  /// The Scalar inputs
  std::vector<const Variable<Scalar> *> _from;

  /// Per-input weights (declared as buffers or parameters per the
  /// `weight_as_parameter` option, mirroring `LinearCombination`)
  std::vector<const Scalar *> _weights;

  /// The exponent \f$ p \f$
  const Scalar & _p;
};
} // namespace neml2
