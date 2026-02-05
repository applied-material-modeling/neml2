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
#include "neml2/misc/types.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/jit.h"

namespace neml2
{
namespace utils
{
static ATensor
pad_prepend(const ATensor & s, Size dim, Size pad)
{
  neml_assert_dbg(s.defined(), "pad_prepend: shape must be defined");
  neml_assert_dbg(s.scalar_type() == kInt64, "pad_prepend: shape must be of type int64");
  neml_assert_dbg(s.dim() == 1, "pad_prepend: shape must be 1D");
  return at::cat({at::full({dim - s.size(0)}, pad, s.options()), s});
}

TraceableTensorShape
broadcast_dynamic_sizes(const std::vector<Tensor> & tensors)
{
  Size dim = 0;
  auto shapes = std::vector<ATensor>{};
  for (const auto & t : tensors)
    if (t.defined())
    {
      dim = t.dynamic_dim() > dim ? t.dynamic_dim() : dim;
      const auto shape = t.dynamic_sizes().as_tensor();
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

TensorShape
broadcast_intmd_sizes(const std::vector<Tensor> & tensors)
{
  Size dim = 0;
  auto shapes = std::vector<TensorShape>{};
  for (const auto & t : tensors)
    if (t.defined())
    {
      dim = t.intmd_dim() > dim ? t.intmd_dim() : dim;
      shapes.emplace_back(t.intmd_sizes());
    }
  if (shapes.empty())
    return {};

  for (auto & s : shapes)
    s = pad_prepend(s, dim, 1);
  auto bshape = TensorShape(dim, 1);

  for (std::size_t i = 0; i < std::size_t(dim); i++)
    for (const auto & s : shapes)
      if (s[i] > bshape[i])
        bshape[i] = s[i];

  return bshape;
}
} // namespace utils

Tensor::Tensor(const ATensor & tensor, Size dynamic_dim, Size intmd_dim)
  : TensorBase(tensor, dynamic_dim, intmd_dim)
{
}

Tensor::Tensor(const ATensor & tensor, const TraceableTensorShape & dynamic_shape, Size intmd_dim)
  : TensorBase(tensor, dynamic_shape, intmd_dim)
{
}

Tensor
Tensor::create(const TensorDataContainer & data, const TensorOptions & options)
{
  return create(data, 0, 0, options);
}

Tensor
Tensor::create(const TensorDataContainer & data,
               Size dynamic_dim,
               Size intmd_dim,
               const TensorOptions & options)
{
  return Tensor(torch::autograd::make_variable(data.convert_to_tensor(options.requires_grad(false)),
                                               options.requires_grad()),
                dynamic_dim,
                intmd_dim)
      .clone(); // clone to take ownership of the data
}

Tensor
Tensor::empty(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::empty(base_shape, options), 0);
}

Tensor
Tensor::empty(const TraceableTensorShape & dynamic_shape,
              TensorShapeRef intmd_shape,
              TensorShapeRef base_shape,
              const TensorOptions & options)
{
  // Record dynamic shape
  if (jit::tracer::isTracing())
    for (Size i = 0; i < (Size)dynamic_shape.size(); ++i)
      if (const auto * const si = dynamic_shape[i].traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "size", dynamic_shape.size() + intmd_shape.size() + base_shape.size(), i, *si);

  return Tensor(
      at::empty(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_shape), options),
      dynamic_shape,
      Size(intmd_shape.size()));
}

Tensor
Tensor::zeros(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::zeros(base_shape, options), 0);
}

Tensor
Tensor::zeros(const TraceableTensorShape & dynamic_shape,
              TensorShapeRef intmd_shape,
              TensorShapeRef base_shape,
              const TensorOptions & options)
{
  // Record dynamic shape
  for (Size i = 0; i < (Size)dynamic_shape.size(); ++i)
    if (const auto * const si = dynamic_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", dynamic_shape.size() + intmd_shape.size() + base_shape.size(), i, *si);

  return Tensor(
      at::zeros(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_shape), options),
      dynamic_shape,
      Size(intmd_shape.size()));
}

Tensor
Tensor::ones(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::ones(base_shape, options), 0);
}

Tensor
Tensor::ones(const TraceableTensorShape & dynamic_shape,
             TensorShapeRef intmd_shape,
             TensorShapeRef base_shape,
             const TensorOptions & options)
{
  // Record dynamic shape
  for (Size i = 0; i < (Size)dynamic_shape.size(); ++i)
    if (const auto * const si = dynamic_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", dynamic_shape.size() + intmd_shape.size() + base_shape.size(), i, *si);

  return Tensor(
      at::ones(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_shape), options),
      dynamic_shape,
      Size(intmd_shape.size()));
}

Tensor
Tensor::full(TensorShapeRef base_shape, const CScalar & init, const TensorOptions & options)
{
  return Tensor(at::full(base_shape, init, options), 0, 0);
}

Tensor
Tensor::full(const TraceableTensorShape & dynamic_shape,
             TensorShapeRef intmd_shape,
             TensorShapeRef base_shape,
             const CScalar & init,
             const TensorOptions & options)
{
  // Record dynamic shape
  for (Size i = 0; i < (Size)dynamic_shape.size(); ++i)
    if (const auto * const si = dynamic_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", dynamic_shape.size() + intmd_shape.size() + base_shape.size(), i, *si);

  return Tensor(
      at::full(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_shape), init, options),
      dynamic_shape,
      Size(intmd_shape.size()));
}

Tensor
Tensor::rand(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(at::rand(base_shape, options), 0);
}

Tensor
Tensor::rand(const TraceableTensorShape & dynamic_shape,
             TensorShapeRef intmd_shape,
             TensorShapeRef base_shape,
             const TensorOptions & options)
{
  // Record dynamic shape
  for (Size i = 0; i < (Size)dynamic_shape.size(); ++i)
    if (const auto * const si = dynamic_shape[i].traceable())
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", dynamic_shape.size() + intmd_shape.size() + base_shape.size(), i, *si);

  return Tensor(
      at::rand(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_shape), options),
      dynamic_shape,
      Size(intmd_shape.size()));
}

Tensor
Tensor::identity(Size n, const TensorOptions & options)
{
  return Tensor(at::eye(n, options), 0);
}
} // end namespace neml2
