// Copyright 2024, UChicago Argonne, LLC
// All Rights Rerved
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

#include "neml2/solvers/LinearSystem.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/norm_sq.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2::es
{
bool
Vector::requires_grad() const
{
  for (const auto & vi : *this)
    if (vi.requires_grad())
      return true;
  return false;
}

Vector
Vector::operator-() const
{
  std::vector<Tensor> result(size());
  for (size_t i = 0; i < size(); i++)
    result[i] = -(*this)[i];
  return Vector(std::move(result));
}

void
Vector::update_data(const Vector & other)
{
  neml_assert_dbg(size() == other.size(),
                  "es::Vector::update_data: size mismatch (",
                  size(),
                  ") vs (",
                  other.size(),
                  ")");
  for (size_t i = 0; i < size(); i++)
  {
    auto & v = (*this)[i];
    v = v.variable_data() + other[i];
  }
}

void
Vector::update(const Vector & other)
{
  neml_assert_dbg(size() == other.size(),
                  "es::Vector::update: size mismatch (",
                  size(),
                  ") vs (",
                  other.size(),
                  ")");
  for (size_t i = 0; i < size(); i++)
  {
    auto & v = (*this)[i];
    v = v + other[i];
  }
}

Scalar
norm_sq(const Vector & v)
{
  Scalar sum_sq = Scalar::zeros();
  for (const auto & vi : v)
    sum_sq += neml2::norm_sq(vi);
  return sum_sq;
}

Scalar
norm(const Vector & v)
{
  return neml2::sqrt(norm_sq(v));
}

} // namespace neml2::es
