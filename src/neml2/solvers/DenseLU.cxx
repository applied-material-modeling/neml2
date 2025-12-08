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
#include "neml2/tensors/functions/linalg/solve.h"
#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/functions/linalg/lu_solve.h"
#include "neml2/tensors/shape_utils.h"
#include <ATen/ops/split.h>

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

es::Vector
DenseLU::solve(const es::Matrix & A, const es::Vector & b) const
{
  // assemble A and b into flat tensors and solve
  const auto [A_f, A_rs, A_cs] = A.assemble();
  const auto [b_f, b_cs] = b.assemble();
  const auto x_f = linalg::solve(A_f, b_f);

  // split the solution back into es::Vector
  const auto & S = x_f.dynamic_sizes();
  auto xfs = at::split(x_f, b_cs, -1);
  std::vector<Tensor> x(xfs.size());
  for (std::size_t i = 0; i < xfs.size(); ++i)
    x[i] = Tensor(xfs[i], S).base_reshape(b.block_sizes(i));
  return es::Vector(std::move(x), b.block_sizes());
}

es::Matrix
DenseLU::solve(const es::Matrix & A, const es::Matrix & B) const
{
  // assemble A and B into flat tensors and solve
  const auto [A_f, A_rs, A_cs] = A.assemble();
  const auto [B_f, B_rs, B_cs] = B.assemble();
  const auto X_f = linalg::solve(A_f, B_f);

  // split the solution back into es::Matrix
  const auto & S = X_f.dynamic_sizes();
  auto xfs = at::split(X_f, A_cs, -2);
  auto X = es::Matrix(A.block_col_sizes(), B.block_col_sizes());
  for (std::size_t i = 0; i < A.n(); ++i)
  {
    auto xs = at::split(xfs[i], B_cs, -1);
    for (std::size_t j = 0; j < B.n(); ++j)
      X(i, j) = Tensor(xs[j], S).base_reshape(
          utils::add_shapes(A.block_col_sizes(i), B.block_col_sizes(j)));
  }
  return X;
}

std::vector<es::Matrix>
DenseLU::solve(const es::Matrix & A, const std::vector<es::Matrix> & B) const
{
  // assemble A and factorize
  const auto [A_f, A_rs, A_cs] = A.assemble();
  const auto [lu, pivot] = linalg::lu_factor(A_f);

  // assemble B into flat tensors and solve
  std::vector<es::Matrix> X(B.size());
  for (std::size_t i = 0; i < B.size(); ++i)
  {
    const auto & Bi = B[i];

    // Xi = 0 if Bi = 0
    if (Bi.zero())
      continue;

    const auto [B_f, B_rs, B_cs] = Bi.assemble();
    const auto X_f = linalg::lu_solve(lu, pivot, B_f);

    // split the solution back into es::Matrix
    const auto & S = X_f.dynamic_sizes();
    auto xfs = at::split(X_f, A_cs, -2);
    auto Xi = es::Matrix(A.block_col_sizes(), Bi.block_col_sizes());
    for (std::size_t i = 0; i < A.n(); ++i)
    {
      auto xs = at::split(xfs[i], B_cs, -1);
      for (std::size_t j = 0; j < Bi.n(); ++j)
        Xi(i, j) = Tensor(xs[j], S).base_reshape(
            utils::add_shapes(A.block_col_sizes(i), Bi.block_col_sizes(j)));
    }
    X[i] = std::move(Xi);
  }

  return X;
}
}
