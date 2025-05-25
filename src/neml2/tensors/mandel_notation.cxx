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

#include "neml2/tensors/mandel_notation.h"
#include "neml2/tensors/TensorCache.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{
const Tensor &
full_to_mandel_map(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({0, 4, 8, 5, 2, 1}, opt); };
  thread_local TensorCache _ftmm(maker);
  return _ftmm(opt);
}

const Tensor &
mandel_to_full_map(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({0, 5, 4, 5, 1, 3, 4, 3, 2}, opt); };
  thread_local TensorCache _mtfm(maker);
  return _mtfm(opt);
}

const Tensor &
full_to_mandel_factor(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2}, opt); };
  thread_local TensorCache _ftmf(maker);
  return _ftmf(opt);
}

const Tensor &
mandel_to_full_factor(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  {
    return Tensor::create(
        {1.0, invsqrt2, invsqrt2, invsqrt2, 1.0, invsqrt2, invsqrt2, invsqrt2, 1.0}, opt);
  };
  thread_local TensorCache _mtff(maker);
  return _mtff(opt);
}

const Tensor &
full_to_skew_map(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor { return Tensor::create({7, 2, 3}, opt); };
  thread_local TensorCache _ftsm(maker);
  return _ftsm(opt);
}

const Tensor &
skew_to_full_map(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({0, 2, 1, 2, 0, 0, 1, 0, 0}, opt); };
  thread_local TensorCache _stfm(maker);
  return _stfm(opt);
}

const Tensor &
full_to_skew_factor(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({1.0, 1.0, 1.0}, opt); };
  thread_local TensorCache _ftsf(maker);
  return _ftsf(opt);
}

const Tensor &
skew_to_full_factor(const TensorOptions & opt)
{
  auto maker = [](const TensorOptions & opt) -> Tensor
  { return Tensor::create({0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0}, opt); };
  thread_local TensorCache _stff(maker);
  return _stff(opt);
}

Tensor
full_to_reduced(const Tensor & full, const Tensor & rmap, const Tensor & rfactors, Size dim)
{
  const auto & batch_shape = full.batch_sizes();
  auto batch_dim = full.batch_dim();
  auto trailing_dim = full.base_dim() - dim - 2; // 2 comes from the reduced axes (3,3)
  auto starting_shape = full.base_sizes().slice(0, dim);
  auto trailing_shape = full.base_sizes().slice(dim + 2);

  indexing::TensorIndices net(dim, indexing::None);
  net.push_back(indexing::Ellipsis);
  net.insert(net.end(), trailing_dim, indexing::None);
  auto map_shape = utils::add_shapes(starting_shape, rmap.size(0), trailing_shape);
  auto map = rmap.index(net).expand(map_shape);
  auto factor = rfactors.index(net);

  auto batched_map = Tensor(map, 0).batch_expand_as(full);
  auto reduced = at::gather(full.base_reshape(utils::add_shapes(starting_shape, 9, trailing_shape)),
                            batch_dim + dim,
                            batched_map);

  return Tensor(factor, 0) * Tensor(reduced, batch_shape);
}

Tensor
reduced_to_full(const Tensor & reduced, const Tensor & rmap, const Tensor & rfactors, Size dim)
{
  const auto & batch_shape = reduced.batch_sizes();
  auto batch_dim = reduced.batch_dim();
  auto trailing_dim = reduced.base_dim() - dim - 1; // There's only 1 axis to unsqueeze
  auto starting_shape = reduced.base_sizes().slice(0, dim);
  auto trailing_shape = reduced.base_sizes().slice(dim + 1);

  indexing::TensorIndices net(dim, indexing::None);
  net.push_back(indexing::Ellipsis);
  net.insert(net.end(), trailing_dim, indexing::None);
  auto map_shape = utils::add_shapes(starting_shape, rmap.size(0), trailing_shape);
  auto map = rmap.index(net).expand(map_shape);
  auto factor = rfactors.index(net);

  auto batched_map = Tensor(map, 0).batch_expand_as(reduced);
  auto full = Tensor(factor * at::gather(reduced, batch_dim + dim, batched_map), batch_shape);

  return full.base_reshape(utils::add_shapes(starting_shape, 3, 3, trailing_shape));
}

Tensor
full_to_mandel(const Tensor & full, Size dim)
{
  return full_to_reduced(full,
                         full_to_mandel_map(full.options().dtype(default_integer_dtype())),
                         full_to_mandel_factor(full.options()),
                         dim);
}

Tensor
mandel_to_full(const Tensor & mandel, Size dim)
{
  return reduced_to_full(mandel,
                         mandel_to_full_map(mandel.options().dtype(default_integer_dtype())),
                         mandel_to_full_factor(mandel.options()),
                         dim);
}

Tensor
full_to_skew(const Tensor & full, Size dim)
{
  return full_to_reduced(full,
                         full_to_skew_map(full.options().dtype(default_integer_dtype())),
                         full_to_skew_factor(full.options()),
                         dim);
}

Tensor
skew_to_full(const Tensor & skew, Size dim)
{
  return reduced_to_full(skew,
                         skew_to_full_map(skew.options().dtype(default_integer_dtype())),
                         skew_to_full_factor(skew.options()),
                         dim);
}
} // namespace neml2
