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

#include <torch/csrc/autograd/variable.h>
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"
#include "neml2/jit/types.h"

namespace neml2
{
namespace utils
{
ATensor
pad_prepend(const ATensor & s, Size dim, Size pad)
{
  neml_assert_dbg(s.defined(), "pad_prepend: shape must be defined");
  neml_assert_dbg(s.scalar_type() == kInt64, "pad_prepend: shape must be of type int64");
  neml_assert_dbg(s.dim() == 1, "pad_prepend: shape must be 1D");
  return at::cat({at::full({dim - s.size(0)}, pad, s.options()), s});
}

TraceableTensorShape
broadcast_batch_sizes(const std::vector<Tensor> & tensors)
{
  Size dim = 0;
  auto shapes = std::vector<ATensor>{};
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
  const auto all_shapes = at::stack(shapes);
  return std::get<0>(at::max(all_shapes, 0));
}
} // namespace utils

Tensor::Tensor(const ATensor & tensor, Size batch_dim)
  : TensorBase(tensor, batch_dim)
{
}

Tensor::Tensor(const ATensor & tensor, const TraceableTensorShape & batch_shape)
  : TensorBase(tensor, batch_shape)
{
}

Tensor
Tensor::create(const TensorDataContainer & data, const TensorOptions & options)
{
  return create(data, 0, options);
}

Tensor
Tensor::create(const TensorDataContainer & data, Size batch_dim, const TensorOptions & options)
{
  return Tensor(torch::autograd::make_variable(data.convert_to_tensor(options.requires_grad(false)),
                                               options.requires_grad()),
                batch_dim);
}

Tensor
Tensor::empty(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::empty(base_shape, options), 0);
}

Tensor
Tensor::empty(const TraceableTensorShape & batch_shape,
              TensorShapeRef base_shape,
              const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(at::empty(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
Tensor::zeros(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::zeros(base_shape, options), 0);
}

Tensor
Tensor::zeros(const TraceableTensorShape & batch_shape,
              TensorShapeRef base_shape,
              const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(at::zeros(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
Tensor::ones(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::ones(base_shape, options), 0);
}

Tensor
Tensor::ones(const TraceableTensorShape & batch_shape,
             TensorShapeRef base_shape,
             const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(at::ones(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
Tensor::full(TensorShapeRef base_shape, Real init, const TensorOptions & options)
{
  return Tensor(at::full(base_shape, init, options), 0);
}

Tensor
Tensor::full(const TraceableTensorShape & batch_shape,
             TensorShapeRef base_shape,
             Real init,
             const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(at::full(utils::add_shapes(batch_shape.concrete(), base_shape), init, options),
                batch_shape);
}

Tensor
Tensor::identity(Size n, const TensorOptions & options)
{
  return Tensor(at::eye(n, options), 0);
}

Tensor
Tensor::identity(const TraceableTensorShape & batch_shape, Size n, const TensorOptions & options)
{
  return identity(n, options).batch_expand_copy(batch_shape);
}

Tensor
Tensor::base_unsqueeze_to(Size n) const
{
  neml_assert_dbg(n >= base_dim(),
                  "base_unsqueeze_to: n (",
                  n,
                  ") must be greater than or equal to base_dim (",
                  base_dim(),
                  ").");
  indexing::TensorIndices net{indexing::Ellipsis};
  net.insert(net.end(), n - base_dim(), indexing::None);
  return Tensor(index(net), batch_sizes());
}
} // end namespace neml2
