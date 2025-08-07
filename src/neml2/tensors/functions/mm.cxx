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

#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/functions/utils.h"

namespace neml2
{
Tensor
mm(const Tensor & a, const Tensor & b)
{
  neml_assert_dynamic_broadcastable_dbg(a, b);
  neml_assert_dbg(a.base_dim() == 2,
                  "The first tensor in bmm has base dimension ",
                  a.base_dim(),
                  " instead of 2.");
  neml_assert_dbg(b.base_dim() == 2,
                  "The second tensor in bmm has base dimension ",
                  b.base_dim(),
                  " instead of 2.");
  const auto [aa, bb, i] = utils::align_intmd_dim(a, b);
  return Tensor(at::matmul(aa, bb), utils::broadcast_dynamic_dim(a, b), i);
}
} // namespace neml2
