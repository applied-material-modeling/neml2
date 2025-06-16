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

#include "neml2/solvers/Newton.h"
#include "neml2/solvers/TrustRegionSubProblem.h"

namespace neml2
{
/**
 * @copydoc neml2::Newton
 *
 * Each update step is obtained by solving a trust-region subproblem, i.e. a quadratic
 * bound-constrained problem.
 *
 * The trust-region outer loop is implemented following Nocedal and Wright, section 4.1.
 *
 * The subproblem implemented here is an alternative linearization of the original subproblem,
 * introduced in
 *
 * > Yuan, Ya-xiang. Trust region algorithms for nonlinear equations. Hong Kong Baptist
 * > University, Department of Mathematics, 1994.
 */
class NewtonWithTrustRegion : public Newton
{
public:
  static OptionSet expected_options();

  NewtonWithTrustRegion(const OptionSet & options);

protected:
  /// Extract options for the subproblem
  OptionSet subproblem_options(const OptionSet &) const;

  /// Extract options for the subproblem solver
  OptionSet subproblem_solver_options(const OptionSet &) const;

  void prepare(const NonlinearSystem & system, const NonlinearSystem::Sol<true> & x) override;

  void update(NonlinearSystem & system,
              NonlinearSystem::Sol<true> & x,
              const NonlinearSystem::Res<true> & r,
              const NonlinearSystem::Jac<true> & J) override;

  NonlinearSystem::Sol<true> solve_direction(const NonlinearSystem::Res<true> & r,
                                             const NonlinearSystem::Jac<true> & J) override;

  /// Reduction in the merit function
  Scalar merit_function_reduction(const NonlinearSystem::Res<true> & r,
                                  const NonlinearSystem::Jac<true> & J,
                                  const NonlinearSystem::Sol<true> & p) const;

  /// Trust-region subproblem
  TrustRegionSubProblem _subproblem;

  /// Solver used to solver the trust-region subproblem
  Newton _subproblem_solver;

  /// The trust region radius
  Scalar _delta;

  /// Initial size of the trust region
  double _delta_0;

  /// Maximum size of the trust region
  double _delta_max;

  /// Criteria for reducing the trust region
  double _reduce_criteria;

  /// Criteria for expanding the trust region
  double _expand_criteria;

  /// Cutback factor if we do reduce the trust region
  double _reduce_factor;

  /// Expansion factor if we do increase the trust region
  double _expand_factor;

  /// Acceptance criteria for a step
  double _accept_criteria;
};
} // namespace neml2
