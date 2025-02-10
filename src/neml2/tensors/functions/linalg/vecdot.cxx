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

#include "neml2/tensors/functions/linalg/vecdot.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/assertions.h"

namespace neml2::linalg
{
Scalar
vecdot(const Tensor & u, const Tensor & v)
{
  neml_assert_batch_broadcastable_dbg(u, v);
  neml_assert_dbg(u.base_dim() == 1,
                  "The first tensor in vecdot has base dimension ",
                  u.base_dim(),
                  " instead of 1.");
  neml_assert_dbg(v.base_dim() == 1,
                  "The second tensor in vecdot has base dimension ",
                  v.base_dim(),
                  " instead of 1.");
  neml_assert_dbg(u.base_size(0) == v.base_size(0),
                  "The two tensors in vecdot have different base sizes: ",
                  u.base_size(0),
                  " and ",
                  v.base_size(0),
                  ".");
  return Scalar(at::linalg_vecdot(u, v));
}
} // namespace neml2::linalg
