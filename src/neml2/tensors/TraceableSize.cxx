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

#include <ATen/ScalarOps.h>

#include "neml2/tensors/TraceableSize.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
const ATensor *
TraceableSize::traceable() const noexcept
{
  return std::get_if<ATensor>(this);
}

Size
TraceableSize::concrete() const
{
  if (const auto * const size = traceable())
  {
    neml_assert_dbg(size->scalar_type() == kInt64, "TraceableSize: size must be of type int64");
    neml_assert_dbg(size->dim() == 0, "TraceableSize: shape must be 0D");
    return size->item<Size>();
  }

  return std::get<Size>(*this);
}

ATensor
TraceableSize::as_tensor() const
{
  if (const auto * const size = traceable())
    return *size;

  return c10::scalar_to_tensor(std::get<Size>(*this));
}

Scalar
TraceableSize::as_scalar() const
{
  return Scalar(as_tensor(), 0);
}

TraceableSize
operator*(const TraceableSize & lhs, const TraceableSize & rhs)
{
  const auto * const l = lhs.traceable();
  const auto * const r = rhs.traceable();
  if (l && r)
    return TraceableSize(*l * *r);
  if (l)
    return TraceableSize(*l * rhs.concrete());
  if (r)
    return TraceableSize(lhs.concrete() * *r);
  return TraceableSize(lhs.concrete() * rhs.concrete());
}

bool
operator==(const TraceableSize & lhs, const TraceableSize & rhs)
{
  return lhs.concrete() == rhs.concrete();
}

bool
operator!=(const TraceableSize & lhs, const TraceableSize & rhs)
{
  return !(lhs == rhs);
}

std::ostream &
operator<<(std::ostream & os, const TraceableSize & s)
{
  os << s.concrete();
  return os;
}
} // namespace neml2
