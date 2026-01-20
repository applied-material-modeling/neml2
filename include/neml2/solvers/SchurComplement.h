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
 * A linear solver which uses the Schur complement method to solve systems with a specific
 * structure.
 *
 * This is particularly useful for systems that can be partitioned into blocks where one block is
 * significantly smaller than the others, and the coupling between blocks is limited.
 *
 * The variables corresponding to the smaller block are referred to as "schur variables", and the
 * rest are "primary variables". The solver first eliminates the primary variables to form a reduced
 * system involving only the schur variables. After solving this reduced system, it back-substitutes
 * to find the primary variables.
 *
 * Consider a linear system represented in block matrix form:
 *
 *   | A11  A12 | | x1 |   =   | b1 |
 *   | A21  A22 | | x2 |       | b2 |
 *
 * Here, x1 represents the primary variables, and x2 represents the schur variables.
 *
 * The solver proceeds as follows:
 * 1. Compute the Schur complement S = A22 - A21 * A11^{-1} * A12.
 * 2. Compute the modified right-hand side c = b2 - A21 * A11^{-1} * b1.
 * 3. Solve the reduced system S * x2 = c for x2.
 * 4. Back-substitute to find x1 using x1 = A11^{-1} * (b1 - A12 * x2).
 */
class SchurComplement : public LinearSolver
{
public:
  static OptionSet expected_options();

  SchurComplement(const OptionSet & options);

  HVector solve(const HMatrix & A, const HVector & b) const override;
  HMatrix solve(const HMatrix & A, const HMatrix & B) const override;
};
} // namespace neml2
