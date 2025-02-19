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

#include "neml2/tensors/functions/mean.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
#define DEFINE_BATCH_MEAN(T)                                                                       \
  T batch_mean(const T & a, Size d)                                                                \
  {                                                                                                \
    neml_assert_dbg(a.batched(), "Must have a batch dimension to take average");                   \
    auto d2 = d >= 0 ? d : d - a.base_dim();                                                       \
    return T(at::mean(a, d2), a.batch_dim() - 1);                                                  \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_BATCH_MEAN);

Tensor
base_mean(const Tensor & a, Size d)
{
  auto d2 = d < 0 ? d : d + a.batch_dim();
  return Tensor(at::mean(a, d2), a.batch_sizes());
}
} // namespace neml2
