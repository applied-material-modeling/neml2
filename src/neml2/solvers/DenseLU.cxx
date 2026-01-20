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
#include "neml2/tensors/Tensor.h"
#include "neml2/equation_systems/LinearSystem.h"
#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/tensors/functions/linalg/solve.h"

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

SparseTensorList
DenseLU::solve(LinearSystem & sys) const
{
  const auto & bilayout = sys.intmd_blayout();
  const auto & uilayout = sys.intmd_ulayout();
  const auto & blayout = sys.blayout();
  const auto & ulayout = sys.ulayout();

  // assemble A and b into flat tensors
  const auto [A, b] = sys.A_and_b();
  const auto Af = assemble(A, bilayout, uilayout, blayout, ulayout);
  const auto bf = assemble(b, bilayout, blayout);

  // solve
  const auto xf = linalg::solve(Af, bf);

  // disassemble the solution
  return disassemble(xf, sys.intmd_ulayout(), sys.ulayout());
}

SparseTensorList
DenseLU::ift(NonlinearSystem & sys) const
{
  const auto & bilayout = sys.intmd_blayout();
  const auto & uilayout = sys.intmd_ulayout();
  const auto & gilayout = sys.intmd_glayout();
  const auto & blayout = sys.blayout();
  const auto & ulayout = sys.ulayout();
  const auto & glayout = sys.glayout();

  // assemble A and B into flat tensors
  const auto Af = assemble(sys.A(), bilayout, uilayout, blayout, ulayout);
  const auto Bf = assemble(sys.B(), bilayout, gilayout, blayout, glayout);

  // solve
  const auto Xf = -linalg::solve(Af, Bf);

  // disassemble the solution
  return disassemble(Xf, uilayout, gilayout, ulayout, glayout);
}

}
