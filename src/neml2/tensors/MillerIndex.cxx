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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/gcd.h"

namespace neml2
{
Scalar
MillerIndex::operator()(Size i) const
{
  return Scalar(base_index({i}), batch_sizes());
}

MillerIndex
MillerIndex::fill(int64_t a, int64_t b, int64_t c, const TensorOptions & options)
{
  return MillerIndex::create({a, b, c}, options);
}

MillerIndex
MillerIndex::reduce() const
{
  auto cf = neml2::gcd(neml2::gcd((*this)(0), (*this)(1)), (*this)(2));
  return *this / cf;
}

Vec
MillerIndex::to_vec(const TensorOptions & options) const
{
  return Vec(this->to(options));
}

Vec
MillerIndex::to_normalized_vec(const TensorOptions & options) const
{
  auto v = this->to_vec(options);
  return v / v.norm();
}
} // namespace neml2
