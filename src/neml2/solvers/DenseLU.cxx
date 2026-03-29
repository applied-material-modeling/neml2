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

#include "neml2/solvers/DenseLU.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/tensors/functions/linalg/solve.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(DenseLU);

OptionSet
DenseLU::expected_options()
{
  OptionSet options = LinearSolver::expected_options();
  options.doc() =
      "Dense LU linear solver. This solver assembles the (possibly) sparse matrix into "
      "a dense one and uses a standard LU decomposition to solve the system of equations.";
  return options;
}

DenseLU::DenseLU(const OptionSet & options)
  : LinearSolver(options)
{
}

SparseVector
DenseLU::solve(const SparseMatrix & A, const SparseVector & b) const
{
  // solve
  const auto xf = linalg::solve(A.assemble(), b.assemble());

  // disassemble the solution
  SparseVector x(A.col_layout, b.istr);
  x.disassemble(xf);
  return x;
}

SparseMatrix
DenseLU::solve(const SparseMatrix & A, const SparseMatrix & B) const
{
  // solve
  const auto Xf = linalg::solve(A.assemble(), B.assemble());

  // disassemble the solution
  SparseMatrix X(A.col_layout, B.col_layout, A.istr);
  X.disassemble(Xf);
  return X;
}

}
