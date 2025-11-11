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

#include "neml2/tensors/functions/linalg/lu_solve.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/utils.h"
#include "neml2/misc/assertions.h"

namespace neml2::linalg
{
Tensor
lu_solve(const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint)
{
  neml_assert_dbg(LU.scalar_type() == neml2::kFloat32 || LU.scalar_type() == neml2::kFloat64,
                  "LU solve only supports float32 and float64, got",
                  LU.scalar_type(),
                  " for factored LU.");
  neml_assert_dbg(B.scalar_type() == neml2::kFloat32 || B.scalar_type() == neml2::kFloat64,
                  "LU solve only supports float32 and float64, got",
                  B.scalar_type(),
                  " for right hand side.");

  neml_assert_dbg(neml2::utils::dynamic_broadcastable(LU, B),
                  "LU and B tensors are not dynamic broadcastable: ",
                  LU.dynamic_sizes(),
                  " vs ",
                  B.dynamic_sizes());
  neml_assert_dbg(neml2::utils::intmd_broadcastable(LU, B),
                  "LU and B tensors are not intmd broadcastable: ",
                  LU.intmd_sizes(),
                  " vs ",
                  B.intmd_sizes());
  neml_assert_dbg(neml2::utils::dynamic_broadcastable(LU, pivots),
                  "LU and pivots tensors are not dynamic broadcastable: ",
                  LU.dynamic_sizes(),
                  " vs ",
                  pivots.dynamic_sizes());
  neml_assert_dbg(neml2::utils::intmd_broadcastable(LU, pivots),
                  "LU and pivots tensors are not intmd broadcastable: ",
                  LU.intmd_sizes(),
                  " vs ",
                  pivots.intmd_sizes());
  neml_assert_dbg(
      LU.base_size(-2) == LU.base_size(-1), "LU tensor is not square: ", LU.base_sizes());
  neml_assert_dbg(LU.base_dim() == 2, "LU tensor does not have base dimension 2: ", LU.base_dim());
  neml_assert_dbg(
      B.base_dim() == 2,
      "B tensor does not have base dimension 2: ",
      B.base_dim(),
      ". Note that torch::linalg::lu_solve requires B to be 2D even for vector right hand sides.");

  auto [aligned_LU, aligned_B, aligned_pivots, i] = utils::align_intmd_dim(LU, B, pivots);

  return Tensor(at::linalg_lu_solve(aligned_LU, aligned_pivots, aligned_B, left, adjoint),
                utils::broadcast_dynamic_dim(LU, B, pivots),
                i);
}
} // namespace neml2::linalg
