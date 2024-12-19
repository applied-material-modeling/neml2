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

#include "neml2/tensors/Scalar.h"
#include "neml2/misc/math.h"

namespace neml2
{

Scalar::Scalar(Real init, const torch::TensorOptions & options)
  : Scalar(Scalar::full(init, options))
{
}

Scalar
Scalar::identity_map(const torch::TensorOptions & options)
{
  return Scalar::ones(options);
}

namespace math
{
Scalar
minimum(const Scalar & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  indexing::TensorIndices net{torch::indexing::Ellipsis};
  net.insert(net.end(), a.base_dim(), torch::indexing::None);
  return Scalar(torch::minimum(a, b.index(net)), broadcast_batch_dim(a, b));
}
}

Scalar
operator*(const Scalar & a, const Scalar & b)
{
  neml_assert_batch_broadcastable_dbg(a, b);
  return torch::operator*(a, b);
}

Scalar
abs(const Scalar & a)
{
  return Scalar(torch::abs(a), a.batch_sizes());
}

namespace math
{
// Scalar
// sigmoid(const Scalar & a, const Scalar & n)
//{
//   neml_assert_broadcastable_dbg(a, n);
//   return 1.0 / 2.0 * (1.0 + math::tanh(n * a));
// }
} // namespace math
} // namespace neml2
