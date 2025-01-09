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

#include "neml2/tensors/MillerIndex.h"
#include "neml2/tensors/Vec.h"

namespace neml2
{
MillerIndex
MillerIndex::fill(Integer a, Integer b, Integer c, const torch::TensorOptions & options)
{
  return torch::tensor({a, b, c}, options);
}

MillerIndex
MillerIndex::reduce() const
{
  // This probably could be vectorized...
  Integer cf =
      torch::gcd(torch::gcd(this->base_index({0}), this->base_index({1})), this->base_index({2}))
          .item<Integer>();
  return *this / cf;
}

Vec
MillerIndex::to_vec(const torch::TensorOptions & options) const
{
  return this->to(options);
}

Vec
MillerIndex::to_normalized_vec(const torch::TensorOptions & options) const
{
  auto v = this->to_vec(options);
  return v / v.norm();
}
} // namespace neml2
