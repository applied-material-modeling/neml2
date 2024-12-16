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

/**
 * This file contains implementation details of the TensorBase class.
 * Refer to `TensorBase.h` for the class definition.
 */

#pragma once

#include "neml2/tensors/TensorBase.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/math.h"
#include "neml2/jit/utils.h"

namespace neml2
{
template <class Derived>
TensorBase<Derived>::TensorBase(const torch::Tensor & tensor, Size batch_dim)
  : torch::Tensor(tensor),
    _batch_dim(batch_dim),
    _batch_sizes(utils::extract_batch_sizes(tensor, batch_dim))
{
  neml_assert_dbg((Size)sizes().size() >= _batch_dim,
                  "Tensor dimension ",
                  sizes().size(),
                  " is smaller than the requested number of batch dimensions ",
                  _batch_dim);
}

template <class Derived>
TensorBase<Derived>::TensorBase(const torch::Tensor & tensor,
                                const TraceableTensorShape & batch_shape)
  : torch::Tensor(tensor),
    _batch_dim(Size(batch_shape.size())),
    _batch_sizes(batch_shape)
{
  neml_assert_dbg(batch_sizes() == batch_shape,
                  "Tensor of shape ",
                  sizes(),
                  " cannot be constructed with batch shape ",
                  batch_shape);
}

template <class Derived>
Derived
TensorBase<Derived>::empty_like(const Derived & other)
{
  return Derived(torch::empty_like(other), other.batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::zeros_like(const Derived & other)
{
  return Derived(torch::zeros_like(other), other.batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::ones_like(const Derived & other)
{
  return Derived(torch::ones_like(other), other.batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::full_like(const Derived & other, Real init)
{
  return Derived(torch::full_like(other, init), other.batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::linspace(const Derived & start, const Derived & end, Size nstep, Size dim)
{
  neml_assert_broadcastable_dbg(start, end);
  neml_assert_dbg(nstep > 0, "nstep must be positive.");

  auto res = start.batch_unsqueeze(dim);

  if (nstep > 1)
  {
    auto Bd = broadcast_batch_dim(start, end);
    auto diff = (end - start).batch_unsqueeze(dim);

    indexing::TensorIndices net(dim, indexing::None);
    net.push_back(indexing::Ellipsis);
    net.insert(net.end(), Bd - dim, indexing::None);
    Scalar steps = torch::arange(nstep, diff.options()).index(net) / (nstep - 1);

    res = res + steps * diff;
  }

  return res;
}

template <class Derived>
Derived
TensorBase<Derived>::logspace(
    const Derived & start, const Derived & end, Size nstep, Size dim, Real base)
{
  auto exponent = neml2::Tensor::linspace(start, end, nstep, dim);
  return math::pow(base, exponent);
}

template <class Derived>
Derived
TensorBase<Derived>::clone(torch::MemoryFormat memory_format) const
{
  return Derived(torch::Tensor::clone(memory_format), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::detach() const
{
  return Derived(torch::Tensor::detach(), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::to(const torch::TensorOptions & options) const
{
  return Derived(torch::Tensor::to(options), batch_sizes());
}

template <class Derived>
bool
TensorBase<Derived>::batched() const
{
  return _batch_dim;
}

template <class Derived>
Size
TensorBase<Derived>::batch_dim() const
{
  return _batch_dim;
}

template <class Derived>
Size
TensorBase<Derived>::base_dim() const
{
  return dim() - batch_dim();
}

template <class Derived>
const TraceableTensorShape &
TensorBase<Derived>::batch_sizes() const
{
  return _batch_sizes;
}

template <class Derived>
TraceableSize
TensorBase<Derived>::batch_size(Size index) const
{
  const auto i = index >= 0 ? index : index + batch_dim();

  // Put the batch size into the traced graph if we are tracing
  if (torch::jit::tracer::isTracing())
    return torch::jit::tracer::getSizeOf(*this, i);

  return size(i);
}

template <class Derived>
TensorShapeRef
TensorBase<Derived>::base_sizes() const
{
  return sizes().slice(_batch_dim);
}

template <class Derived>
Size
TensorBase<Derived>::base_size(Size index) const
{
  return base_sizes()[index >= 0 ? index : index + base_dim()];
}

template <class Derived>
Size
TensorBase<Derived>::base_storage() const
{
  return utils::storage_size(base_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), torch::indexing::Slice());
  auto res = this->index(indices_vec);
  return Derived(res, res.dim() - base_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices2(batch_dim(), torch::indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  return neml2::Tensor(this->index(indices2), batch_sizes());
}

template <class Derived>
void
TensorBase<Derived>::batch_index_put_(indexing::TensorIndicesRef indices,
                                      const torch::Tensor & other)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), torch::indexing::Slice());
  this->index_put_(indices_vec, other);
}

template <class Derived>
void
TensorBase<Derived>::batch_index_put_(indexing::TensorIndicesRef indices, Real v)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), torch::indexing::Slice());
  this->index_put_(indices_vec, v);
}

template <class Derived>
void
TensorBase<Derived>::base_index_put_(indexing::TensorIndicesRef indices,
                                     const torch::Tensor & other)
{
  indexing::TensorIndices indices2(batch_dim(), torch::indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  this->index_put_(indices2, other);
}

template <class Derived>
void
TensorBase<Derived>::base_index_put_(indexing::TensorIndicesRef indices, Real v)
{
  indexing::TensorIndices indices2(batch_dim(), torch::indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  this->index_put_(indices2, v);
}

template <class Derived>
Derived
TensorBase<Derived>::variable_data() const
{
  return Derived(torch::Tensor::variable_data(), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_expand(const TraceableTensorShape & batch_shape) const
{
  // We don't want to touch the base dimensions, so put -1 for them.
  auto net = batch_shape.concrete();
  net.insert(net.end(), base_dim(), -1);

  // Record the batch sizes in the traced graph if we are tracing
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem("size", net.size(), i, *si);

  return Derived(expand(net), batch_shape);
}

template <class Derived>
Derived
TensorBase<Derived>::batch_expand(const TraceableSize & batch_size, Size dim) const
{
  // We don't want to touch other batch dimensions and the base dimensions, so put -1 for them.
  auto net = std::vector<Size>(this->dim(), -1);
  auto i = dim >= 0 ? dim : this->dim() + dim - base_dim();
  net[i] = batch_size.concrete();

  // Record the batch sizes in the traced graph if we are tracing
  if (const auto * const s = batch_size.traceable())
    torch::jit::tracer::ArgumentStash::stashIntArrayRefElem("size", this->dim(), i, *s);

  return Derived(expand(net), batch_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand(TensorShapeRef base_shape) const
{
  if (base_sizes() == base_shape)
    return *this;

  // We don't want to touch the batch dimensions, so put -1 for them.
  auto net = base_shape.vec();
  net.insert(net.begin(), batch_dim(), -1);
  return neml2::Tensor(expand(net), batch_sizes());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand(Size base_size, Size dim) const
{
  if (this->base_size(dim) == base_size)
    return *this;

  // We don't want to touch the batch dimensions and other base dimensions, so put -1 for them.
  auto net = std::vector<Size>(this->dim(), -1);
  auto i = dim < 0 ? this->dim() + dim : dim + batch_dim();
  net[i] = base_size;
  return neml2::Tensor(expand(net), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_expand_copy(const TraceableTensorShape & batch_shape) const
{
  return Derived(batch_expand(batch_shape).contiguous(), batch_shape);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand_copy(TensorShapeRef base_shape) const
{
  return neml2::Tensor(base_expand(base_shape).contiguous(), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_reshape(const TraceableTensorShape & batch_shape) const
{
  // Record the batch sizes in the traced graph if we are tracing
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "shape", batch_shape.size() + base_dim(), i, *si);

  return Derived(reshape(utils::add_shapes(batch_shape.concrete(), base_sizes())), batch_shape);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_reshape(TensorShapeRef base_shape) const
{
  auto batch_shape = batch_sizes();

  // Record the batch sizes in the traced graph if we are tracing
  for (Size i = 0; i < (Size)batch_shape.size(); ++i)
    if (const auto * const si = batch_shape[i].traceable())
      torch::jit::tracer::ArgumentStash::stashIntArrayRefElem(
          "shape", batch_shape.size() + base_shape.size(), i, *si);

  return neml2::Tensor(reshape(utils::add_shapes(batch_shape.concrete(), base_shape)),
                       batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_unsqueeze(Size d) const
{
  auto d2 = d >= 0 ? d : d - base_dim();
  return Derived(unsqueeze(d2), _batch_dim + 1);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_unsqueeze(Size d) const
{
  auto d2 = d < 0 ? d : d + batch_dim();
  return neml2::Tensor(torch::Tensor::unsqueeze(d2), batch_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_transpose(Size d1, Size d2) const
{
  return Derived(
      torch::Tensor::transpose(d1 < 0 ? d1 - base_dim() : d1, d2 < 0 ? d2 - base_dim() : d2),
      _batch_dim);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_transpose(Size d1, Size d2) const
{
  return neml2::Tensor(
      torch::Tensor::transpose(d1 < 0 ? d1 : _batch_dim + d1, d2 < 0 ? d2 : _batch_dim + d2),
      batch_sizes());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_flatten() const
{
  if (base_dim() == 1)
    return *this;

  return base_reshape({base_storage()});
}

template <class Derived>
Derived
TensorBase<Derived>::operator-() const
{
  return Derived(-torch::Tensor(*this), batch_sizes());
}

} // end namespace neml2
