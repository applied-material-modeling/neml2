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
#include "neml2/tensors/ConstantTensors.h"

namespace neml2
{
Tensor
full_to_reduced(const Tensor & full, const ATensor & rmap, const ATensor & rfactors, Size dim)
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
  auto factor = rfactors.to(full).index(net);

  auto batched_map = Tensor(map, 0).batch_expand_as(full);
  auto reduced = at::gather(full.base_reshape(utils::add_shapes(starting_shape, 9, trailing_shape)),
                            batch_dim + dim,
                            batched_map);

  return Tensor(factor, 0) * Tensor(reduced, batch_shape);
}

Tensor
reduced_to_full(const Tensor & reduced, const ATensor & rmap, const ATensor & rfactors, Size dim)
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
  auto factor = rfactors.to(reduced).index(net);

  auto batched_map = Tensor(map, 0).batch_expand_as(reduced);
  auto full = Tensor(factor * at::gather(reduced, batch_dim + dim, batched_map), batch_shape);

  return full.base_reshape(utils::add_shapes(starting_shape, 3, 3, trailing_shape));
}

Tensor
full_to_mandel(const Tensor & full, Size dim)
{
  return full_to_reduced(
      full,
      ConstantTensors::full_to_mandel_map().to(full.options().dtype(default_integer_dtype())),
      ConstantTensors::full_to_mandel_factor().to(full.options()),
      dim);
}

Tensor
mandel_to_full(const Tensor & mandel, Size dim)
{
  return reduced_to_full(
      mandel,
      ConstantTensors::mandel_to_full_map().to(mandel.options().dtype(default_integer_dtype())),
      ConstantTensors::mandel_to_full_factor().to(mandel.options()),
      dim);
}

Tensor
full_to_skew(const Tensor & full, Size dim)
{
  return full_to_reduced(
      full,
      ConstantTensors::full_to_skew_map().to(full.options().dtype(default_integer_dtype())),
      ConstantTensors::full_to_skew_factor().to(full.options()),
      dim);
}

Tensor
skew_to_full(const Tensor & skew, Size dim)
{
  return reduced_to_full(
      skew,
      ConstantTensors::skew_to_full_map().to(skew.options().dtype(default_integer_dtype())),
      ConstantTensors::skew_to_full_factor().to(skew.options()),
      dim);
}
} // namespace neml2
