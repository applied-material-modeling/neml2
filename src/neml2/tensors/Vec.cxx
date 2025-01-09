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

#include "neml2/tensors/Vec.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"

#include "neml2/tensors/functions/linalg/vecdot.h"
#include "neml2/tensors/functions/linalg/cross.h"
#include "neml2/tensors/functions/linalg/outer.h"

namespace neml2
{
Vec::Vec(const Rot & r)
  : Vec(Tensor(r))
{
}

R2
Vec::identity_map(const TensorOptions & options)
{
  return R2::identity(options);
}

Scalar
Vec::dot(const Vec & v) const
{
  return linalg::vecdot(*this, v);
}

Vec
Vec::cross(const Vec & v) const
{
  return linalg::cross(*this, v);
}

R2
Vec::outer(const Vec & v) const
{
  return linalg::outer(*this, v);
}

Vec
Vec::transform(const R2 & op) const
{
  return op * (*this);
}

} // namespace neml2
