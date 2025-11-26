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

#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/misc/assertions.h"

namespace neml2::linalg
{
std::tuple<Tensor, Tensor>
lu_factor(const Tensor & A, bool pivot)
{
  neml_assert_dbg(A.scalar_type() == neml2::kFloat32 || A.scalar_type() == neml2::kFloat64,
                  "LU factorization only supports float32 and float64, got ",
                  A.scalar_type());
  neml_assert_dbg(A.base_size(-2) == A.base_size(-1), "A tensor is not square: ", A.base_sizes());
  neml_assert_dbg(A.base_dim() == 2, "A tensor does not have base dimension 2: ", A.base_dim());

  auto [LU, pivots] = at::linalg_lu_factor(A, pivot);
  return {Tensor(LU, A.dynamic_sizes(), A.intmd_dim()),
          Tensor(pivots, A.dynamic_sizes(), A.intmd_dim())};
}
} // namespace neml2::linalg
