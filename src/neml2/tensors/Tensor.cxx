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

#pragma once

#include <c10/core/ScalarTypeToTypeMeta.h>

#include "neml2/tensors/Tensor.h"
#include "neml2/jit/utils.h"
#include "neml2/tensors/assertions.h"
#include "neml2/tensors/utils.h"

namespace neml2
{
Tensor::Tensor(const at::Tensor & tensor, Size batch_dim)
  : at::Tensor(tensor),
    _batch_sizes(utils::trace_leading_sizes(tensor, batch_dim))
{
}

Tensor::Tensor(at::Tensor && tensor, Size batch_dim)
  : at::Tensor(std::move(tensor)),
    _batch_sizes(utils::trace_leading_sizes(tensor, batch_dim))
{
}

Tensor::Tensor(const at::Tensor & tensor, const TraceableTensorShape & batch_shape)
  : at::Tensor(tensor),
    _batch_sizes(batch_shape)
{
  neml_assert_dbg(batch_sizes() == batch_shape,
                  "Tensor of shape ",
                  tensor.sizes(),
                  " cannot be constructed with batch shape ",
                  batch_shape,
                  ". Leading dimensions must match.");
}

Tensor::Tensor(at::Tensor && tensor, TraceableTensorShape && batch_shape)
  : at::Tensor(std::move(tensor)),
    _batch_sizes(std::move(batch_shape))
{
  neml_assert_dbg(batch_sizes() == batch_shape,
                  "Tensor of shape ",
                  tensor.sizes(),
                  " cannot be constructed with batch shape ",
                  batch_shape,
                  ". Leading dimensions must match.");
}

Tensor::Tensor(const Tensor & tensor)
  : at::Tensor(tensor),
    _batch_sizes(tensor.batch_sizes())
{
}

Tensor::Tensor(Tensor && tensor) noexcept
  : at::Tensor(std::move(tensor)),
    _batch_sizes(std::move(tensor._batch_sizes))
{
}

Tensor &
Tensor::operator=(const Tensor & tensor)
{
  at::Tensor::operator=(tensor);
  _batch_sizes = tensor.batch_sizes();
  return *this;
}

Tensor &
Tensor::operator=(Tensor && tensor) noexcept
{
  at::Tensor::operator=(std::move(tensor));
  _batch_sizes = std::move(tensor._batch_sizes);
  return *this;
}

Tensor
Tensor::to(const TensorOptions & options) const
{
  return Tensor(std::move(at::Tensor::to(options)), std::move(_batch_sizes));
}

Tensor
Tensor::clone() const
{
  return Tensor(std::move(at::Tensor::clone()), std::move(_batch_sizes));
}

Tensor
Tensor::detach() const
{
  return Tensor(std::move(at::Tensor::detach()), std::move(_batch_sizes));
}

Tensor &
Tensor::detach_()
{
  at::Tensor::detach_();
  return *this;
}

Tensor &
Tensor::copy_(const at::Tensor & other)
{
  at::Tensor::copy_(other);
  return *this;
}

Tensor &
Tensor::zero_()
{
  at::Tensor::zero_();
  return *this;
}

Tensor &
Tensor::requires_grad_(bool req)
{
  at::Tensor::requires_grad_(req);
  return *this;
}

bool
Tensor::batched() const
{
  return !_batch_sizes.empty();
}

Size
Tensor::batch_dim() const
{
  return Size(_batch_sizes.size());
}

Size
Tensor::base_dim() const
{
  return dim() - batch_dim();
}

const TraceableTensorShape &
Tensor::batch_sizes() const
{
  return _batch_sizes;
}

TraceableSize
Tensor::batch_size(Size dim) const
{
  dim = utils::bound_dim(dim, 0, batch_dim());
  return is_tracing() ? trace_size(*this, dim) : size(dim);
}

TensorShapeRef
Tensor::base_sizes() const
{
  return sizes().slice(batch_dim());
}

Size
Tensor::base_size(Size dim) const
{
  dim = utils::bound_dim(dim, batch_dim(), dim());
  return size(dim);
}

Size
Tensor::base_storage() const
{
  return utils::storage_size(base_sizes());
}

Tensor
Tensor::batch_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  auto res = this->index(indices_vec);
  return Tensor(res, res.dim() - base_dim());
}

Tensor
Tensor::base_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  return Tensor(this->index(indices2), batch_sizes());
}

Tensor
Tensor::batch_slice(Size dim, const indexing::Slice & index) const
{
  dim = utils::bound_dim(dim, 0, batch_dim());
  auto res = at::Tensor::slice(
      i, index.start().expect_int(), index.stop().expect_int(), index.step().expect_int());
  return Tensor(res, res.dim() - base_dim());
}

Tensor
Tensor::base_slice(Size dim, const indexing::Slice & index) const
{
  dim = utils::bound_dim(dim, batch_dim(), dim());
  auto res = at::Tensor::slice(
      i, index.start().expect_int(), index.stop().expect_int(), index.step().expect_int());
  return Tensor(res, batch_sizes());
}

Tensor &
Tensor::batch_index_put_(indexing::TensorIndicesRef indices, const at::Tensor & other)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  at::Tensor::index_put_(indices_vec, other);
  return *this;
}

Tensor &
Tensor::batch_index_put_(indexing::TensorIndicesRef indices, const LScalar & v)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  at::Tensor::index_put_(indices_vec, v);
  return *this;
}

Tensor &
Tensor::base_index_put_(indexing::TensorIndicesRef indices, const at::Tensor & other)
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  at::Tensor::index_put_(indices2, other);
  return *this;
}

Tensor &
Tensor::base_index_put_(indexing::TensorIndicesRef indices, const LScalar & v)
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  at::Tensor::index_put_(indices2, v);
  return *this;
}

Tensor
Tensor::variable_data() const
{
  return Tensor(std::move(variable_data()), std::move(_batch_sizes));
}

Tensor
Tensor::batch_expand(const TraceableTensorShape & batch_shape) const
{
  // We don't want to touch the base dimensions, so put -1 for them.
  auto net = batch_shape.concrete();
  net.insert(net.end(), base_dim(), -1);

  // Record the batch sizes in the traced graph if we are tracing
  if (is_tracing())
    for (Size i = 0; i < (Size)batch_shape.size(); ++i)
      if (const auto * const si = batch_shape[i].traceable())
        TracerArgumentStash::stashIntArrayRefElem("size", net.size(), i, *si);

  return Tensor(std::move(expand(net)), batch_shape);
}

Tensor
Tensor::batch_expand(const TraceableSize & batch_size, Size dim) const
{
  // We don't want to touch other batch dimensions and the base dimensions, so put -1 for them.
  auto net = std::vector<Size>(this->dim(), -1);
  dim = utils::bound_dim(dim, 0, batch_dim());
  net[dim] = batch_size.concrete();

  // Record the batch sizes in the traced graph if we are tracing
  if (is_tracing())
    if (const auto * const s = batch_size.traceable())
      TracerArgumentStash::stashIntArrayRefElem("size", this->dim(), dim, *s);

  return Tensor(std::move(expand(net)), batch_dim());
}

Tensor
Tensor::base_expand(TensorShapeRef base_shape) const
{
  if (base_sizes() == base_shape)
    return *this;

  // We don't want to touch the batch dimensions, so put -1 for them.
  auto net = base_shape.vec();
  net.insert(net.begin(), batch_dim(), -1);
  return Tensor(std::move(expand(net)), std::move(_batch_sizes));
}

Tensor
Tensor::base_expand(Size base_size, Size dim) const
{
  if (this->base_size(dim) == base_size)
    return *this;

  // We don't want to touch the batch dimensions and other base dimensions, so put -1 for them.
  auto net = std::vector<Size>(this->dim(), -1);
  dim = utils::bound_dim(dim, batch_dim(), dim());
  net[dim] = base_size;
  return Tensor(std::move(expand(net)), std::move(_batch_sizes));
}

Tensor
Tensor::batch_expand_as(const Tensor & other) const
{
  return batch_expand(other.batch_sizes());
}

Tensor
Tensor::base_expand_as(const Tensor & other) const
{
  return base_expand(other.base_sizes());
}

Tensor
Tensor::batch_expand_copy(const TraceableTensorShape & batch_shape) const
{
  return batch_expand(batch_shape).clone();
}

Tensor
Tensor::base_expand_copy(TensorShapeRef base_shape) const
{
  return base_expand(base_shape).clone();
}

Tensor
Tensor::batch_reshape(const TraceableTensorShape & batch_shape) const
{
  // Record the batch sizes in the traced graph if we are tracing
  if (is_tracing())
    for (Size i = 0; i < (Size)batch_shape.size(); ++i)
      if (const auto * const si = batch_shape[i].traceable())
        TracerArgumentStash::stashIntArrayRefElem("shape", batch_shape.size() + base_dim(), i, *si);

  return Tensor(std::move(reshape(utils::add_shapes(batch_shape.concrete(), base_sizes()))),
                batch_shape);
}

Tensor
Tensor::base_reshape(TensorShapeRef base_shape) const
{
  // Record the batch sizes in the traced graph if we are tracing
  if (is_tracing())
    for (Size i = 0; i < batch_dim(); ++i)
      if (const auto * const si = batch_size(i).traceable())
        TracerArgumentStash::stashIntArrayRefElem("shape", batch_dim() + base_shape.size(), i, *si);

  return Tensor(std::move(reshape(utils::add_shapes(batch_sizes().concrete(), base_shape))),
                std::move(_batch_sizes));
}

Tensor
Tensor::batch_unsqueeze(Size d) const
{
  d = utils::bound_dim(d, 0, batch_dim());
  auto new_batch_sizes = _batch_sizes;
  new_batch_sizes.insert(new_batch_sizes.begin() + d, 1);
  return Tensor(std::move(unsqueeze(d)), std::move(new_batch_sizes));
}

Tensor
Tensor::base_unsqueeze(Size d) const
{
  d = utils::bound_dim(d, batch_dim(), dim());
  return Tensor(std::move(unsqueeze(d)), std::move(_batch_sizes));
}

Tensor
Tensor::batch_transpose(Size d1, Size d2) const
{
  d1 = utils::bound_dim(d1, 0, batch_dim());
  d2 = utils::bound_dim(d2, 0, batch_dim());
  auto new_batch_sizes = _batch_sizes;
  std::iter_swap(new_batch_sizes.begin() + d1, new_batch_sizes.begin() + d2);
  return Tensor(std::move(transpose(d1, d2)), std::move(new_batch_sizes));
}

Tensor
Tensor::base_transpose(Size d1, Size d2) const
{
  d1 = utils::bound_dim(d1, batch_dim(), dim());
  d2 = utils::bound_dim(d2, batch_dim(), dim());
  return Tensor(std::move(transpose(d1, d2)), std::move(_batch_sizes));
}

Tensor
Tensor::base_flatten() const
{
  if (base_dim() == 1)
    return *this;

  return base_reshape({base_storage()});
}

Tensor
Tensor::operator~() const
{
  return Tensor(std::move(bitwise_not()), std::move(_batch_sizes));
}

Tensor
Tensor::operator-() const
{
  return Tensor(std::move(neg()), std::move(_batch_sizes));
}

Tensor &
operator+=(const Tensor & other)
{
  add_(other);
  return *this;
}

Tensor &
operator+=(const LScalar & other)
{
  add_(other);
  return *this;
}

Tensor &
operator-=(const Tensor & other)
{
  sub_(other);
  return *this;
}

Tensor &
operator-=(const LScalar & other)
{
  sub_(other);
  return *this;
}

Tensor &
operator*=(const Tensor & other)
{
  mul_(other);
  return *this;
}

Tensor &
operator*=(const LScalar & other)
{
  mul_(other);
  return *this;
}

Tensor &
operator/=(const Tensor & other)
{
  div_(other);
  return *this;
}

Tensor &
operator/=(const LScalar & other)
{
  div_(other);
  return *this;
}

Tensor &
operator&=(const Tensor & other)
{
  bitwise_and_(other);
  return *this;
}

Tensor &
operator|=(const Tensor & other)
{
  bitwise_or_(other);
  return *this;
}

Tensor &
operator^=(const Tensor & other)
{
  bitwise_xor_(other);
  return *this;
}

Tensor
operator==(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::eq(a, b)), a.batch_sizes());
}

Tensor
operator==(const LScalar & a, const Tensor & b)
{
  return b == a;
}

Tensor
operator==(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::eq(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator!=(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::ne(a, b)), a.batch_sizes());
}

Tensor
operator!=(const LScalar & a, const Tensor & b)
{
  return b != a;
}

Tensor
operator!=(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::ne(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator>=(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::ge(a, b)), a.batch_sizes());
}

Tensor
operator>=(const LScalar & a, const Tensor & b)
{
  return b <= a;
}

Tensor
operator>=(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::ge(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator<=(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::le(a, b)), a.batch_sizes());
}

Tensor
operator<=(const LScalar & a, const Tensor & b)
{
  return b >= a;
}

Tensor
operator<=(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::le(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator>(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::gt(a, b)), a.batch_sizes());
}

Tensor
operator>(const LScalar & a, const Tensor & b)
{
  return b < a;
}

Tensor
operator>(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::gt(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator<(const Tensor & a, const LScalar & b)
{
  return Tensor(std::move(at::lt(a, b)), a.batch_sizes());
}

Tensor
operator<(const LScalar & a, const Tensor & b)
{
  return b > a;
}

Tensor
operator<(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(std::move(at::lt(a, b)), utils::broadcast_batch_dim(a, b));
}

Tensor
operator+(const Tensor & a, const LScalar & b)
{
  return Tensor(at::add(a, b), a.batch_sizes());
}

Tensor
operator+(const LScalar & a, const Tensor & b)
{
  return b + a;
}

Tensor
operator+(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(at::add(a, b), utils::broadcast_batch_dim(a, b));
}

Tensor
operator-(const Tensor & a, const LScalar & b)
{
  return Tensor(at::sub(a, b), a.batch_sizes());
}

Tensor
operator-(const LScalar & a, const Tensor & b)
{
  return -b + a;
}

Tensor
operator-(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(at::sub(a, b), utils::broadcast_batch_dim(a, b));
}

Tensor
operator*(const Tensor & a, const LScalar & b)
{
  return Tensor(at::mul(a, b), a.batch_sizes());
}

Tensor
operator*(const LScalar & a, const Tensor & b)
{
  return b * a;
}

Tensor
operator*(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(at::mul(a, b), utils::broadcast_batch_dim(a, b));
}

Tensor
operator/(const Tensor & a, const LScalar & b)
{
  return Tensor(at::div(a, b), a.batch_sizes());
}

Tensor
operator/(const LScalar & a, const Tensor & b)
{
  return Tensor(at::mul(b.reciprocal(), a), b.batch_sizes());
}

Tensor
operator/(const Tensor & a, const Tensor & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Tensor(at::div(a, b), utils::broadcast_batch_dim(a, b));
}
} // namespace neml2
