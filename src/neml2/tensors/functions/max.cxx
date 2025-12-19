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

#include "neml2/tensors/functions/max.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{
#define DEFINE_MAX(T)                                                                              \
  T dynamic_max(const T & a, Size d)                                                               \
  {                                                                                                \
    d = utils::normalize_dim(d, 0, a.dynamic_dim());                                               \
    auto sizes = a.dynamic_sizes();                                                                \
    sizes.erase(sizes.begin() + d);                                                                \
    return T(std::get<0>(at::max(a, d)), sizes, a.intmd_dim());                                    \
  }                                                                                                \
  T intmd_max(const T & a, Size d)                                                                 \
  {                                                                                                \
    d = utils::normalize_dim(d, a.dynamic_dim(), a.batch_dim());                                   \
    return T(std::get<0>(at::max(a, d)), a.dynamic_sizes(), a.intmd_dim() - 1);                    \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_MAX);

Tensor
base_max(const Tensor & a, Size d)
{
  d = utils::normalize_dim(d, a.batch_dim(), a.dim());
  return Tensor(std::get<0>(at::max(a, d)), a.dynamic_sizes(), a.intmd_dim());
}
} // namespace neml2
