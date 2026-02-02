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

#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/sqrt.h"

namespace neml2
{
SparseTensorList::SparseTensorList(const std::vector<Tensor> & v)
  : std::vector<Tensor>(v)
{
}

TensorOptions
SparseTensorList::options() const
{
  for (const auto & t : *this)
    if (t.defined())
      return t.options();
  throw NEMLException("Cannot determine options of empty SparseTensorList");
}

SparseTensorList
SparseTensorList::data() const
{
  SparseTensorList d(size());
  for (std::size_t i = 0; i < size(); i++)
    if ((*this)[i].defined())
      d[i] = (*this)[i].variable_data();
  return d;
}

SparseTensorList
operator-(const SparseTensorList & a)
{
  SparseTensorList b(a.size());
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i].defined())
      b[i] = -a[i];
  return b;
}

SparseTensorList
operator+(const SparseTensorList & a, const SparseTensorList & b)
{
  neml_assert(a.size() == b.size(),
              "Incompatible sizes in SparseTensorList addition, got ",
              a.size(),
              " and ",
              b.size());
  SparseTensorList c(a.size());
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i].defined() && b[i].defined())
      c[i] = a[i] + b[i];
    else if (a[i].defined())
      c[i] = a[i];
    else if (b[i].defined())
      c[i] = b[i];
  return c;
}

SparseTensorList
operator*(const Scalar & s, const SparseTensorList & a)
{
  SparseTensorList b(a.size());
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i].defined())
      b[i] = s * a[i];
  return b;
}

SparseTensorList
operator*(const SparseTensorList & a, const Scalar & s)
{
  return s * a;
}

Scalar
inner(const SparseTensorList & a, const SparseTensorList & b)
{
  neml_assert(a.size() == b.size(),
              "Incompatible sizes in SparseTensorList inner product, got ",
              a.size(),
              " and ",
              b.size());
  auto s = Scalar::zeros(a.options());
  for (std::size_t i = 0; i < a.size(); i++)
    if (a[i].defined() && b[i].defined())
      s = s + neml2::inner(a[i].static_flatten(), b[i].static_flatten());
  return s;
}

Scalar
norm_sq(const SparseTensorList & a)
{
  return inner(a, a);
}

Scalar
norm(const SparseTensorList & a)
{
  return neml2::sqrt(norm_sq(a));
}

}
