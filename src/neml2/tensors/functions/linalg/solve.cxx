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
solve(const Tensor & A, const Tensor & B, bool check_errors)
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

  // at::linalg_solve interprets B as batched vectors when B.ndim == A.ndim - 1.
  // The best way to handle this is to always make sure A and B are aligned along batch dimensions
  auto [aligned_A, aligned_B, i] = utils::align_intmd_dim(A, B);
  auto bdim_delta = aligned_A.dynamic_dim() - aligned_B.dynamic_dim();
  if (bdim_delta > 0)
    aligned_B = aligned_B.dynamic_unsqueeze(0, bdim_delta);
  else if (bdim_delta < 0)
    aligned_A = aligned_A.dynamic_unsqueeze(0, -bdim_delta);

  // Unfortunately, torch uses a stupid rule to disambiguate vector and matrix RHS. For example,
  // with A of shape (;;3,3) and B of shape (2,1;;3)
  //   - after alignment, we get aligned_A of shape (1,1;;3,3) and aligned_B of shape (2,1;;3)
  //   - at::linalg_solve sees aligned_A as (1,1,3,3) and aligned_B as (2,1,3)
  //   - it tries to interpret aligned_B as batched vectors, however the batch shape of aligned_A
  //     (1,1) is not the same as the batch shape of aligned_B (2,1), so it decides to interpret
  //     aligned_B as batched matrices instead.
  //   - next, since matrix shape of A is (3,3) and matrix shape of B is (1,3), it complains about
  //     incompatible matrix shapes.
  //
  // This is a really bad logic, but I'm not bothered to argue with it. The solution is to make B a
  // matrix on our side. Only then torch handles batch shape broadcasting correctly.
  if (B.base_dim() == 1)
    aligned_B = aligned_B.base_unsqueeze(1);

  auto [x, info] = at::linalg_solve_ex(
      aligned_A.contiguous(), aligned_B.contiguous(), /*left=*/true, /*check_errors=*/check_errors);

  // Emit error message. This is LAPACK-stye error code, so info == 0 means success, and info > 0
  // means failure.
  if (check_errors)
  {
    auto nfail = (info != 0).sum().item<int>();
    neml_assert(nfail == 0,
                "Failed to solve linear system. ",
                nfail,
                " out of ",
                info.numel(),
                " batches failed to factorize. This is likely due to an ill-conditioned system.");
  }

  // Remove the extra matrix dimension if the original B is a vector
  if (B.base_dim() == 1)
    x = x.squeeze(-1);

  return Tensor(x, utils::broadcast_dynamic_dim(aligned_A, aligned_B), i);
}
} // namespace neml2::linalg
