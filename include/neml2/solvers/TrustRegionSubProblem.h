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

#include "neml2/solvers/NonlinearSystem.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * The trust region subproblem introduced in
 *
 * > Yuan, Ya-xiang. Trust region algorithms for nonlinear equations. Hong Kong Baptist
 * > University, Department of Mathematics, 1994.
 */
class TrustRegionSubProblem : public NonlinearSystem
{
public:
  TrustRegionSubProblem(const OptionSet & options);

  /// Record the current state of the underlying problem
  void reinit(const RES<true> & r, const JAC<true> & J, const Scalar & delta);

  Tensor preconditioned_direction(const Scalar & s) const;

protected:
  void set_guess(const SOL<false> & x) override;

  void assemble(RES<false> * residual, JAC<false> * Jacobian) override;

  Tensor preconditioned_solve(const Scalar & s, const Tensor & v) const;

private:
  /// Solution to the Lagrange multiplier
  Scalar _s;

  /// The trust region radius
  Scalar _delta;

  /// Temporary Jacobian-Jacobian product
  Tensor _JJ;

  /// Temporary Jacobian-Residual product
  Tensor _Jr;
};
} // namespace neml2
