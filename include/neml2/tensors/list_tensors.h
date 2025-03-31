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

#pragma once

#include "neml2/tensors/Tensor.h"

namespace neml2
{
/// @brief outer product on lists, where the first input is a list tensor
template <typename F, typename T1, typename T2>
Tensor
list_derivative_outer_product_a(F && f, const T1 & a, const T2 & b)
{
  return Tensor(std::forward<F>(f)(a, b.batch_unsqueeze(-1)), b.batch_sizes());
}

/// @brief outer product on lists, where the second input is a list tensor
template <typename F, typename T1, typename T2>
Tensor
list_derivative_outer_product_b(F && f, const T1 & a, const T2 & b)
{
  return Tensor(std::forward<F>(f)(a.batch_unsqueeze(-1), b), a.batch_sizes())
      .base_transpose(0, a.base_dim());
}

/// @brief outer product on lists where both inputs are list tensors
template <typename F, typename T1, typename T2>
Tensor
list_derivative_outer_product_ab(F && f, const T1 & a, const T2 & b)
{
  return Tensor(std::forward<F>(f)(a.batch_unsqueeze(-1), b.batch_unsqueeze(-2)), a.batch_dim() - 1)
      .base_transpose(1, 1 + a.base_dim());
}

} // namespace neml2
