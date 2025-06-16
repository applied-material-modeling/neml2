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

#include "neml2/solvers/Solver.h"
#include "neml2/solvers/NonlinearSystem.h"

namespace neml2
{
/**
 * @brief The nonlinear solver solves a nonlinear system of equations.
 *
 */
class NonlinearSolver : public Solver
{
public:
  enum class RetCode : std::uint8_t
  {
    /// Solver converged successfully
    SUCCESS = 0,
    /// Maximum number of iterations reached (without convergence)
    MAXITER = 1,
    /// Solver failed to converge
    FAILURE = 2
  };

  struct Result
  {
    /// Solver return code, @see neml2::NonlinearSolver::RetCode
    RetCode ret = RetCode::FAILURE;
    /// Solution to the nonlinear system
    NonlinearSystem::Sol<false> solution;
    /// Number of iterations before convergence
    std::size_t iterations = 0;
  };

public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new NonlinearSolver object
   *
   * @param options The options extracted from the input file
   */
  NonlinearSolver(const OptionSet & options);

  /**
   * @brief Solve the given nonlinear system.
   *
   * @param system The nonlinear system of equations.
   * @param x0 The initial guess
   * @return @see neml2::NonlinearSolver::Result
   */
  virtual Result solve(NonlinearSystem & system, const NonlinearSystem::Sol<false> & x0) = 0;

  /// Absolute tolerance
  double atol;
  /// Relative tolerance
  double rtol;
  /// Maximum number of iterations
  unsigned int miters;
};
} // namespace neml2
