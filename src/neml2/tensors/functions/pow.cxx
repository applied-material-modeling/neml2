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

#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/assertions.h"

namespace neml2
{
#define DEFINE_POW(T)                                                                              \
  T pow(const T & a, const Real & n) { return T(at::pow(a, n), a.batch_sizes()); }                 \
  T pow(const T & a, const Scalar & n)                                                             \
  {                                                                                                \
    neml_assert_batch_broadcastable_dbg(a, n);                                                     \
    return T(at::pow(a, n.base_unsqueeze_to(a.base_dim())), utils::broadcast_batch_dim(a, n));     \
  }                                                                                                \
  static_assert(true)
FOR_ALL_NONSCALAR_TENSORBASE(DEFINE_POW);

Scalar
pow(const Scalar & a, const Real & n)
{
  return Scalar(at::pow(a, n), a.batch_sizes());
}

Scalar
pow(const Scalar & a, const Scalar & n)
{
  neml_assert_batch_broadcastable_dbg(a, n);
  return Scalar(at::pow(a, n));
}

Scalar
pow(const Real & a, const Scalar & n)
{
  return Scalar(at::pow(a, n), n.batch_sizes());
}

Tensor
pow(const Tensor & a, const Tensor & n)
{
  neml_assert_batch_broadcastable_dbg(a, n);
  neml_assert_dbg(utils::sizes_broadcastable(a.base_sizes(), n.base_sizes()),
                  "Cannot broadcast tensors in pow");
  return Tensor(at::pow(a, n), utils::broadcast_batch_dim(a, n));
}

Tensor
pow(const Real & a, const Tensor & n)
{
  return Tensor(at::pow(a, n), n.batch_sizes());
}
} // namespace neml2
