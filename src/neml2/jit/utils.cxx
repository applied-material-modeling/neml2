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

#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/jit/utils.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
bool
is_tracing()
{
  return torch::jit::tracer::isTracing();
}

void
neml_assert_tracing()
{
  neml_assert(is_tracing(), "Expected to be tracing but not tracing");
}

void
neml_assert_not_tracing()
{
  neml_assert(!is_tracing(), "Tracing is prohibited in this region");
}

void
neml_assert_tracing_dbg()
{
#ifndef NDEBUG
  neml_assert_tracing();
#endif
}

void
neml_assert_not_tracing_dbg()
{
#ifndef NDEBUG
  neml_assert_not_tracing();
#endif
}

namespace utils
{
TraceableTensorShape
extract_leading_sizes(const torch::Tensor & tensor, Size dim)
{
  TraceableTensorShape sizes;
  for (Size i = 0; i < dim; ++i)
    sizes.emplace_back(extract_size(tensor, i));
  return sizes;
}

TraceableSize
extract_size(const torch::Tensor & tensor, Size dim)
{
  neml_assert_tracing_dbg();
  neml_assert_dbg(dim >= 0, "Requested dimension is out of bounds: ", dim, " < 0");
  neml_assert_dbg(
      tensor.dim() >= dim, "Requested dimension is out of bounds: ", dim, " >= ", tensor.dim());

  return torch::jit::tracer::getSizeOf(tensor, dim);
}

torch::Tensor
pad_prepend(const torch::Tensor & s, Size dim, Size pad)
{
  neml_assert_dbg(s.defined(), "pad_prepend: shape must be defined");
  neml_assert_dbg(s.scalar_type() == torch::kInt64, "pad_prepend: shape must be of type int64");
  neml_assert_dbg(s.dim() == 1, "pad_prepend: shape must be 1D");
  return torch::cat({torch::full({dim - s.size(0)}, pad, s.options()), s});
}

TraceableTensorShape
broadcast_batch_sizes(const std::vector<Tensor> & tensors)
{
  Size dim = 0;
  auto shapes = std::vector<torch::Tensor>{};
  for (const auto & t : tensors)
    if (t.defined())
    {
      dim = t.batch_dim() > dim ? t.batch_dim() : dim;
      const auto shape = t.batch_sizes().as_tensor();
      if (shape.defined())
        shapes.push_back(shape);
    }
  if (shapes.empty())
    return TraceableTensorShape(TensorShape{});
  /// Pre-pad ones to the shapes
  for (auto & s : shapes)
    s = pad_prepend(s, dim, 1);
  /// Braodcast
  const auto all_shapes = torch::stack(shapes);
  return std::get<0>(torch::max(all_shapes, 0));
}
} // namespace utils
} // namespace neml2
