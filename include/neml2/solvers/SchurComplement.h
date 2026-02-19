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
 * @brief A linear solver using the Schur complement method.
 *
 * This solver partitions the linear system into two groups of variables (primary and Schur)
 * and applies the Schur complement method to solve the system efficiently.
 *
 * Given a partitioned system:
 * @code
 * [A11 A12] [u1]   [b1]
 * [A21 A22] [u2] = [b2]
 * @endcode
 *
 * Where u1 are the primary unknowns and u2 are the Schur unknowns, the method:
 * 1. Computes the Schur complement: S = A22 - A21 * A11^{-1} * A12
 * 2. Solves for u2: S * u2 = b2 - A21 * A11^{-1} * b1
 * 3. Back-solves for u1: A11 * u1 = b1 - A12 * u2
 *
 * The primary and Schur groups are defined by the equation system variable groups.
 * Each group can have its own linear solver, allowing different solution strategies
 * for different parts of the system.
 *
 * @note This solver requires exactly two variable groups.
 */
class SchurComplement : public LinearSolver
{
public:
  static OptionSet expected_options();

  SchurComplement(const OptionSet & options);

  void setup() override;

  SparseTensorList solve(LinearSystem &) const override;
  SparseTensorList ift(NonlinearSystem &) const override;

private:
  /**
   * @brief Extract a sub-block from a dense matrix.
   *
   * @param A The full matrix
   * @param row_start Starting row index
   * @param row_size Number of rows to extract
   * @param col_start Starting column index
   * @param col_size Number of columns to extract
   * @return Tensor The extracted sub-block
   */
  static Tensor extract_block(const Tensor & A,
                              Size row_start,
                              Size row_size,
                              Size col_start,
                              Size col_size);

  /**
   * @brief Extract a sub-vector from a dense vector.
   *
   * @param b The full vector
   * @param start Starting index
   * @param size Number of elements to extract
   * @return Tensor The extracted sub-vector
   */
  static Tensor extract_subvector(const Tensor & b, Size start, Size size);

  /**
   * @brief Compute the total flattened size for a group of variables.
   *
   * @param intmd_shapes Intermediate shapes
   * @param base_shapes Base shapes
   * @return Size The total flattened size
   */
  static Size compute_group_size(const std::vector<TensorShape> & intmd_shapes,
                                 const std::vector<TensorShape> & base_shapes);

  /// Index of the primary variable group (unknowns for first solve)
  unsigned int _primary_group;

  /// Index of the Schur variable group (unknowns computed via Schur complement)
  unsigned int _schur_group;

  /// Solver for the primary system (A11 * x = b)
  std::shared_ptr<LinearSolver> _primary_solver;

  /// Solver for the Schur complement system (S * x = b)
  std::shared_ptr<LinearSolver> _schur_solver;
};

} // namespace neml2
