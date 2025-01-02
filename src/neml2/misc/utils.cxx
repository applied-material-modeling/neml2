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

#include "neml2/misc/utils.h"
#include <cxxabi.h>
#include <torch/csrc/jit/frontend/tracer.h>

namespace neml2::utils
{
std::string
demangle(const char * name)
{
  // c10 already has an implementation, let's not reinvent the wheels
  return c10::demangle(name);
}

TraceableTensorShape
extract_batch_sizes(const torch::Tensor & tensor, Size batch_dim)
{
  // Put the batch sizes into the traced graph if we are tracing
  // TODO: This could be optimized
  if (torch::jit::tracer::isTracing())
  {
    TraceableTensorShape sizes;
    for (Size i = 0; i < batch_dim; ++i)
      sizes.emplace_back(torch::jit::tracer::getSizeOf(tensor, i));
    return sizes;
  }

  return tensor.sizes().slice(0, batch_dim);
}

Size
storage_size(TensorShapeRef shape)
{
  Size sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<>());
}

TensorShape
pad_prepend(TensorShapeRef s, Size dim, Size pad)
{
  TensorShape s2(s);
  s2.insert(s2.begin(), dim - s.size(), pad);
  return s2;
}

torch::Tensor
pad_prepend(const torch::Tensor & s, Size dim, Size pad)
{
  neml_assert_dbg(s.defined(), "pad_prepend: shape must be defined");
  neml_assert_dbg(s.scalar_type() == torch::kInt64, "pad_prepend: shape must be of type int64");
  neml_assert_dbg(s.dim() == 1, "pad_prepend: shape must be 1D");
  return torch::cat({torch::full({dim - s.size(0)}, pad, s.options()), s});
}

// LCOV_EXCL_START
std::string
indentation(int level, int indent)
{
  std::stringstream ss;
  std::string space(indent, ' ');
  for (int i = 0; i < level; i++)
    ss << space;
  return ss.str();
}
// LCOV_EXCL_STOP

} // namespace neml2
