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

#include "neml2/tensors/functions/normalize_gcd.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
#define DEFINE_NORMALIZE_GCD(T)                                                                    \
  T normalize_gcd(const T & a)                                                                     \
  {                                                                                                \
    auto ai = c10::isIntegralType(a.scalar_type(), false) ? a : a.to(get_default_integer_dtype()); \
    auto af = ai.base_flatten();                                                                   \
                                                                                                   \
    if (af.base_size(0) == 1)                                                                      \
      return Scalar::ones_like(a);                                                                 \
                                                                                                   \
    auto d = at::gcd(af.base_index({0}), af.base_index({1}));                                      \
    for (Size i = 2; i < af.base_size(0); i++)                                                     \
      d = at::gcd(d, af.base_index({i}));                                                          \
                                                                                                   \
    return ai / Scalar(d, a.dynamic_sizes(), a.intmd_dim());                                       \
  }                                                                                                \
  static_assert(true)
FOR_ALL_TENSORBASE(DEFINE_NORMALIZE_GCD);
} // namespace neml2
