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

#include "neml2/solvers/LinearSolver.h"

namespace neml2
{
/**
 * @brief Linear solver using the Schur complement factorization.
 *
 * Solves a block-partitioned system where the system matrix A has at least
 * two variable groups.  Let p denote the primary group index and s the Schur
 * group index.  The block system is
 *
 *   [ A_pp  A_ps ] [ x_p ]   [ b_p ]
 *   [ A_sp  A_ss ] [ x_s ] = [ b_s ]
 *
 * The Schur complement factorization proceeds as
 *
 *   Y   = A_pp^{-1} A_ps           (primary_solver)
 *   z   = A_pp^{-1} b_p            (primary_solver)
 *   S   = A_ss - A_sp Y            (dense arithmetic)
 *   d   = b_s - A_sp z             (dense arithmetic)
 *   x_s = S^{-1} d                 (schur_solver)
 *   x_p = z - Y x_s                (dense arithmetic)
 *
 * The same factorization is used for the matrix RHS overload (AX = B).
 */
class SchurComplement : public LinearSolver
{
public:
  static OptionSet expected_options();

  SchurComplement(const OptionSet & options);

  SparseVector solve(const SparseMatrix &, const SparseVector &) const override;
  SparseMatrix solve(const SparseMatrix &, const SparseMatrix &) const override;

private:
  /// Index of the primary variable group
  const std::size_t _primary_group;
  /// Index of the Schur complement variable group
  const std::size_t _schur_group;
  /// Linear solver for the primary block A_pp
  std::shared_ptr<LinearSolver> _primary_solver;
  /// Linear solver for the Schur complement block S
  std::shared_ptr<LinearSolver> _schur_solver;
};
} // namespace neml2
