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

#include <ATen/ops/stack.h>

#include "neml2/jit/TraceableTensorShape.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
TraceableTensorShape::TraceableTensorShape(const TensorShape & shape)
{
  for (const auto & size : shape)
    emplace_back(size);
}

TraceableTensorShape::TraceableTensorShape(TensorShapeRef shape)
{
  for (const auto & size : shape)
    emplace_back(size);
}

TraceableTensorShape::TraceableTensorShape(Size shape)
  : TraceableTensorShape(TensorShapeRef({shape}))
{
}

TraceableTensorShape::TraceableTensorShape(const ATensor & shape)
{
  neml_assert_dbg(shape.dim() == 1, "TraceableTensorShape: shape must be 1D");
  neml_assert_dbg(shape.scalar_type() == kInt64,
                  "TraceableTensorShape: shape must be of type int64");
  for (Size i = 0; i < shape.size(0); i++)
    emplace_back(shape[i]);
}

TraceableTensorShape
TraceableTensorShape::slice(Size start, Size end) const
{
  if (start < 0)
    start += Size(size());
  if (end < 0)
    end += Size(size());

  return TraceableTensorShape(begin() + start, begin() + end);
}

TraceableTensorShape
TraceableTensorShape::slice(Size N) const
{
  if (N < 0)
    N += Size(size());
  return TraceableTensorShape(begin() + N, end());
}

TensorShape
TraceableTensorShape::concrete() const
{
  TensorShape s;
  for (const auto & size : *this)
    s.push_back(size.concrete());
  return s;
}

ATensor
TraceableTensorShape::as_tensor() const
{
  if (empty())
    return ATensor();

  auto sizes = std::vector<ATensor>(size());
  for (std::size_t i = 0; i < size(); i++)
    sizes[i] = at(i).as_tensor();
  return at::stack(sizes);
}

bool
operator==(const TraceableTensorShape & lhs, const TraceableTensorShape & rhs)
{
  return lhs.concrete() == rhs.concrete();
}

bool
operator!=(const TraceableTensorShape & lhs, const TraceableTensorShape & rhs)
{
  return !(lhs == rhs);
}
} // namespace neml2
