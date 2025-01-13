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

#include "neml2/tensors/factory_methods.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/jit/utils.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/utils.h"

namespace neml2
{
Tensor
empty_like(const Tensor & other)
{
  return Tensor(torch::empty_like(other), other.batch_sizes());
}

Tensor
zeros_like(const Tensor & other)
{
  return Tensor(torch::zeros_like(other), other.batch_sizes());
}

Tensor
ones_like(const Tensor & other)
{
  return Tensor(torch::ones_like(other), other.batch_sizes());
}

Tensor
full_like(const Tensor & other, const NScalar & init)
{
  return Tensor(torch::full_like(other, init), other.batch_sizes());
}

Tensor
empty(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(torch::empty(base_shape, options), 0);
}

Tensor
empty(const TraceableTensorShape & batch_shape,
      TensorShapeRef base_shape,
      const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(torch::empty(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
zeros(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(torch::zeros(base_shape, options), 0);
}

Tensor
zeros(const TraceableTensorShape & batch_shape,
      TensorShapeRef base_shape,
      const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(torch::zeros(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
ones(TensorShapeRef base_shape, const TensorOptions & options)
{
  return Tensor(torch::ones(base_shape, options), 0);
}

Tensor
ones(const TraceableTensorShape & batch_shape,
     TensorShapeRef base_shape,
     const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(torch::ones(utils::add_shapes(batch_shape.concrete(), base_shape), options),
                batch_shape);
}

Tensor
full(TensorShapeRef base_shape, const NScalar & init, const TensorOptions & options)
{
  return Tensor(torch::full(base_shape, init, options), 0);
}

Tensor
full(const TraceableTensorShape & batch_shape,
     TensorShapeRef base_shape,
     const NScalar & init,
     const TensorOptions & options)
{
  // Record batch shape
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "size", batch_shape.size() + base_shape.size(), i, *si);

  return Tensor(torch::full(utils::add_shapes(batch_shape.concrete(), base_shape), init, options),
                batch_shape);
}

Tensor
identity(Size n, const TensorOptions & options)
{
  return Tensor(torch::eye(n, options), 0);
}

Tensor
identity(const TraceableTensorShape & batch_shape, Size n, const TensorOptions & options)
{
  return identity(n, options).batch_expand_copy(batch_shape);
}

Tensor
linspace(const Tensor & start, const Tensor & end, Size nstep, Size dim)
{
  neml_assert_broadcastable_dbg(start, end);
  neml_assert_dbg(nstep > 0, "nstep must be positive.");

  auto res = start.batch_unsqueeze(dim);

  if (nstep > 1)
  {
    auto Bd = utils::broadcast_batch_dim(start, end);
    auto diff = (end - start).batch_unsqueeze(dim);

    indexing::TensorIndices net(dim, indexing::None);
    net.push_back(indexing::Ellipsis);
    net.insert(net.end(), Bd - dim, indexing::None);
    Scalar steps = torch::arange(nstep, diff.options()).index(net) / (nstep - 1);

    res = res + steps * diff;
  }

  return res;
}

Tensor
logspace(const Tensor & start, const Tensor & end, Size nstep, Size dim, Real base)
{
  auto exponent = Tensor::linspace(start, end, nstep, dim);
  return Tensor(torch::pow(base, exponent), exponent.batch_sizes());
}
} // namespace neml2
