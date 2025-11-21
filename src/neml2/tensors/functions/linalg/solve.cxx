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

#include "neml2/tensors/functions/linalg/solve.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/utils.h"

namespace neml2::linalg
{
Tensor
solve(const Tensor & A, const Tensor & B)
{
  neml_assert_dbg(A.scalar_type() == neml2::kFloat32 || A.scalar_type() == neml2::kFloat64,
                  "LU solve only supports float32 and float64, got",
                  A.scalar_type(),
                  " for A.");
  neml_assert_dbg(B.scalar_type() == neml2::kFloat32 || B.scalar_type() == neml2::kFloat64,
                  "LU solve only supports float32 and float64, got",
                  B.scalar_type(),
                  " for right hand side.");

  neml_assert_dbg(neml2::utils::dynamic_broadcastable(A, B),
                  "A and B tensors are not dynamic broadcastable: ",
                  A.dynamic_sizes(),
                  " vs ",
                  B.dynamic_sizes());
  neml_assert_dbg(neml2::utils::intmd_broadcastable(A, B),
                  "A and B tensors are not intmd broadcastable: ",
                  A.intmd_sizes(),
                  " vs ",
                  B.intmd_sizes());
  neml_assert_dbg(A.base_size(-2) == A.base_size(-1), "A tensor is not square: ", A.base_sizes());
  neml_assert_dbg(A.base_dim() == 2, "A tensor does not have base dimension 2: ", A.base_dim());

  // This is obnoxious
  if (B.base_dim() == 1)
  {
    auto [aligned_A, aligned_B, i] = utils::align_intmd_dim(A, B.base_unsqueeze(-1));
    return Tensor(at::linalg_solve(aligned_A.contiguous(), aligned_B.contiguous(), /*left=*/true),
                  utils::broadcast_dynamic_dim(aligned_A, aligned_B),
                  i)
        .base_squeeze(-1);
  }
  auto [aligned_A, aligned_B, i] = utils::align_intmd_dim(A, B);
  return Tensor(at::linalg_solve(aligned_A.contiguous(), aligned_B.contiguous(), /*left=*/true),
                utils::broadcast_dynamic_dim(aligned_A, aligned_B),
                i);
}
} // namespace neml2::linalg
