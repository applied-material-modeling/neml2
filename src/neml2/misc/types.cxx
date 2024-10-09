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

#include "neml2/misc/types.h"
#include "neml2/misc/error.h"

namespace neml2
{
const torch::Tensor *
TraceableSize::traceable() const noexcept
{
  return std::get_if<torch::Tensor>(this);
}

Size
TraceableSize::concrete() const
{
  if (const auto * const size = traceable())
  {
    neml_assert_dbg(size->scalar_type() == torch::kInt64,
                    "TraceableSize: size must be of type int64");
    neml_assert_dbg(size->dim() == 0, "TraceableSize: shape must be 0D");
    return size->item<Size>();
  }

  return std::get<Size>(*this);
}

torch::Tensor
TraceableSize::as_tensor() const
{
  if (const auto * const size = traceable())
    return *size;

  return torch::tensor(std::get<Size>(*this), torch::kInt64);
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

TraceableTensorShape::TraceableTensorShape(const torch::Tensor & shape)
{
  neml_assert_dbg(shape.dim() == 1, "TraceableTensorShape: shape must be 1D");
  neml_assert_dbg(shape.scalar_type() == torch::kInt64,
                  "TraceableTensorShape: shape must be of type int64");
  for (Size i = 0; i < shape.size(0); i++)
    emplace_back(shape.index({i}));
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

torch::Tensor
TraceableTensorShape::as_tensor() const
{
  if (empty())
    return torch::Tensor();

  auto sizes = std::vector<torch::Tensor>(size());
  for (std::size_t i = 0; i < size(); i++)
    sizes[i] = at(i).as_tensor();
  return torch::stack(sizes);
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

torch::TensorOptions &
default_tensor_options()
{
  static torch::TensorOptions _default_tensor_options =
      torch::TensorOptions().dtype(default_dtype()).device(default_device());
  return _default_tensor_options;
}

torch::TensorOptions &
default_integer_tensor_options()
{
  static torch::TensorOptions _default_integer_tensor_options =
      torch::TensorOptions().dtype(default_integer_dtype()).device(default_device());
  return _default_integer_tensor_options;
}

torch::Dtype &
default_dtype()
{
  static torch::Dtype _default_dtype = torch::kFloat64;
  return _default_dtype;
}

torch::Dtype &
default_integer_dtype()
{
  static torch::Dtype _default_integer_dtype = torch::kInt64;
  return _default_integer_dtype;
}

torch::Device &
default_device()
{
  static torch::Device _default_device = torch::kCPU;
  return _default_device;
}

Real &
machine_precision()
{
  static Real _machine_precision = 1E-15;
  return _machine_precision;
}

Real &
tolerance()
{
  static Real _tolerance = 1E-6;
  return _tolerance;
}

Real &
tighter_tolerance()
{
  static Real _tighter_tolerance = 1E-12;
  return _tighter_tolerance;
}

std::string &
buffer_name_separator()
{
  static std::string _buffer_sep = ".";
  return _buffer_sep;
}

std::string &
parameter_name_separator()
{
  static std::string _param_sep = ".";
  return _param_sep;
}

bool &
currently_solving_nonlinear_system()
{
  static bool _solving_nl_sys = false;
  return _solving_nl_sys;
}

bool &
currently_requesting_AD()
{
  static bool _requesting_AD = false;
  return _requesting_AD;
}
} // namespace neml2
