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
#include "neml2/misc/assertions.h"

namespace neml2::linalg
{
Tensor
lu_solve(const Tensor & LU, const Tensor & pivots, const Tensor & B, bool left, bool adjoint)
{
  neml_assert_dbg(
      LU.intmd_dim() == 0, "Intermediate dimension of LU must be 0, got", LU.intmd_dim());
  neml_assert_dbg(pivots.intmd_dim() == 0,
                  "Intermediate dimension of pivots must be 0, got",
                  pivots.intmd_dim());
  neml_assert_dbg(B.intmd_dim() == 0, "Intermediate dimension of B must be 0, got", B.intmd_dim());
  return Tensor(at::linalg_lu_solve(LU, pivots, B, left, adjoint), B.dynamic_sizes(), 0);
}
} // namespace neml2::linalg
