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

namespace neml2
{
class Tensor;
struct HVector;
struct HMatrix;

/**
 * @brief The linear solver solves a linear system of equations.
 *
 */
class LinearSolver : public Solver
{
public:
  static OptionSet expected_options();

  LinearSolver(const OptionSet & options);

  /// Solve Ax = b for x
  virtual HVector solve(const HMatrix & A, const HVector & b) const = 0;

  /// Solve AX = B for X
  virtual HMatrix solve(const HMatrix & A, const HMatrix & B) const = 0;

  ///@{
  /// Whether the derived solver supports LU factorization (and its reuse)
  virtual bool support_lu_factorization() const { return false; }
  /// LU factorization of A, @return tuple of (LU, pivot)
  virtual std::tuple<Tensor, Tensor> lu_factor(const HMatrix & A) const;
  /// LU solve using precomputed LU factors
  virtual HVector lu_solve(const Tensor & LU, const Tensor & pivot, const HVector & b) const;
  /// LU solve using precomputed LU factors
  virtual HMatrix lu_solve(const Tensor & LU, const Tensor & pivot, const HMatrix & B) const;
  ///@}
};
} // namespace neml2
