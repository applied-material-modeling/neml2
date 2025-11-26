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

#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/functions/utils.h"

namespace neml2
{
#define DEFINE_WHERE(T)                                                                            \
  T where(const Tensor & c, const T & a, const T & b)                                              \
  {                                                                                                \
    neml_assert_broadcastable_dbg(a, b);                                                           \
    neml_assert(c.base_sizes() == a.base_sizes() && c.base_sizes() == b.base_sizes(),              \
                "Condition tensor must have the same base shape as both input tensors.");          \
                                                                                                   \
    const auto [cc, aa, bb, i] = utils::align_intmd_dim(c, a, b);                                  \
    return T(at::where(cc, aa, bb), c.dynamic_sizes(), i);                                         \
  }                                                                                                \
                                                                                                   \
  T where(const Scalar & c, const T & a, const T & b)                                              \
  {                                                                                                \
    neml_assert_broadcastable_dbg(a, b);                                                           \
    const auto [cc, aa, bb, i] = utils::align_intmd_dim(c, a, b);                                  \
    return T(at::where(cc.base_unsqueeze(0, a.base_dim()), aa, bb), c.dynamic_sizes(), i);         \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_WHERE);
} // namespace neml2
