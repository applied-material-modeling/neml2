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

#include <ATen/ops/split.h>

#include "neml2/solvers/DenseLU.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/functions/linalg/solve.h"
#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/functions/linalg/lu_solve.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"

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

static HVector
split_vector(const Tensor & x, const std::vector<Size> & s, const std::vector<TensorShapeRef> & bs)
{
  const auto & D = x.dynamic_sizes();
  auto xfs = at::split(x, s, -1);
  std::vector<Tensor> v(xfs.size());
  for (std::size_t i = 0; i < xfs.size(); ++i)
    v[i] = Tensor(xfs[i], D).base_reshape(bs[i]);
  return HVector(std::move(v), bs);
}

static HMatrix
split_matrix(const Tensor & X,
             const std::vector<Size> & rs,
             const std::vector<Size> & cs,
             const std::vector<TensorShapeRef> & rbs,
             const std::vector<TensorShapeRef> & cbs)
{
  const auto & S = X.dynamic_sizes();
  auto xfs = at::split(X, rs, -2);
  auto M = HMatrix(rbs, cbs);
  for (std::size_t i = 0; i < rs.size(); ++i)
  {
    auto xs = at::split(xfs[i], cs, -1);
    for (std::size_t j = 0; j < cs.size(); ++j)
      M(i, j) = Tensor(xs[j], S).base_reshape(utils::add_shapes(rbs[i], cbs[j]));
  }
  return M;
}

HVector
DenseLU::solve(const HMatrix & A, const HVector & b) const
{
  // assemble A and b into flat tensors and solve
  const auto [A_f, A_rs, A_cs] = A.assemble();
  const auto [b_f, b_cs] = b.assemble();
  const auto x_f = linalg::solve(A_f, b_f);

  // split the solution back into HVector
  return split_vector(x_f, b_cs, b.block_sizes());
}

HMatrix
DenseLU::solve(const HMatrix & A, const HMatrix & B) const
{
  // assemble A and B into flat tensors and solve
  const auto [A_f, A_rs, A_cs] = A.assemble();
  const auto [B_f, B_rs, B_cs] = B.assemble();
  const auto X_f = linalg::solve(A_f, B_f);

  // split the solution back into HMatrix
  return split_matrix(X_f, B_rs, B_cs, B.block_row_sizes(), B.block_col_sizes());
}

std::tuple<Tensor, Tensor>
DenseLU::lu_factor(const HMatrix & A) const
{
  // assemble A and factorize
  const auto [A_f, A_rs, A_cs] = A.assemble();
  return linalg::lu_factor(A_f);
}

HVector
DenseLU::lu_solve(const Tensor & LU, const Tensor & pivot, const HVector & b) const
{
  // assemble b into a flat vector and solve
  const auto [b_f, b_cs] = b.assemble();
  const auto x_f = linalg::lu_solve(LU, pivot, b_f);

  // split the solution back into HVector
  return split_vector(x_f, b_cs, b.block_sizes());
}

HMatrix
DenseLU::lu_solve(const Tensor & LU, const Tensor & pivot, const HMatrix & B) const
{
  // assemble B into a flat matrix and solve
  const auto [B_f, B_rs, B_cs] = B.assemble();
  const auto X_f = linalg::lu_solve(LU, pivot, B_f);

  // split the solution back into HVector
  return split_matrix(X_f, B_rs, B_cs, B.block_row_sizes(), B.block_col_sizes());
}

}
