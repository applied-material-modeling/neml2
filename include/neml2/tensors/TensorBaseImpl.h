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

#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/tensors/TraceableTensorShape.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/TensorBase.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/jit.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/assertions.h"

namespace neml2::jit
{
using namespace torch::jit;
}

namespace neml2
{
template <class Derived>
TensorBase<Derived>::TensorBase(const ATensor & tensor, Size dynamic_dim, Size intmd_dim)
  : ATensor(tensor),
    _dynamic_sizes(utils::extract_traceable_sizes(tensor, 0, dynamic_dim)),
    _intmd_dim(intmd_dim)
{
  validate_shapes_and_dims();
}

template <class Derived>
TensorBase<Derived>::TensorBase(const ATensor & tensor,
                                TraceableTensorShape dynamic_shape,
                                Size intmd_dim)
  : ATensor(tensor),
    _dynamic_sizes(std::move(dynamic_shape)),
    _intmd_dim(intmd_dim)
{
  validate_shapes_and_dims();
}

template <class Derived>
void
TensorBase<Derived>::validate_shapes_and_dims() const
{
  neml_assert(dim() >= dynamic_dim() + intmd_dim(),
              "Tensor dimension ",
              dim(),
              " is not sufficient for the requested number of dynamic dimensions (",
              dynamic_dim(),
              ") and intermediate dimensions (",
              intmd_dim(),
              ")");
  neml_assert(dynamic_sizes() == sizes().slice(0, dynamic_dim()),
              "Tensor of shape ",
              sizes(),
              " is incompatible with dynamic shape ",
              dynamic_sizes(),
              ". The leading dimensions must match.");
}

template <class Derived>
Derived
TensorBase<Derived>::empty_like(const Derived & other)
{
  return Derived(at::empty_like(other), other.dynamic_sizes(), other.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::zeros_like(const Derived & other)
{
  return Derived(at::zeros_like(other), other.dynamic_sizes(), other.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::ones_like(const Derived & other)
{
  return Derived(at::ones_like(other), other.dynamic_sizes(), other.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::full_like(const Derived & other, const CScalar & init)
{
  return Derived(at::full_like(other, init), other.dynamic_sizes(), other.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::rand_like(const Derived & other)
{
  return Derived(at::rand_like(other), other.dynamic_sizes(), other.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::contiguous() const
{
  return Derived(ATensor::contiguous(), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::clone() const
{
  return Derived(ATensor::clone(), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::detach() const
{
  return Derived(ATensor::detach(), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::to(const TensorOptions & options) const
{
  return Derived(ATensor::to(options), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Size
TensorBase<Derived>::batch_dim() const
{
  return dynamic_dim() + intmd_dim();
}

template <class Derived>
Size
TensorBase<Derived>::base_dim() const
{
  return dim() - batch_dim();
}

template <class Derived>
Size
TensorBase<Derived>::dynamic_dim() const
{
  return static_cast<Size>(_dynamic_sizes.size());
}

template <class Derived>
Size
TensorBase<Derived>::static_dim() const
{
  return dim() - dynamic_dim();
}

template <class Derived>
Size
TensorBase<Derived>::intmd_dim() const
{
  return _intmd_dim;
}

template <class Derived>
TraceableTensorShape
TensorBase<Derived>::batch_sizes() const
{
  return utils::add_traceable_shapes(dynamic_sizes(), intmd_sizes());
}

template <class Derived>
TensorShapeRef
TensorBase<Derived>::base_sizes() const
{
  return sizes().slice(batch_dim());
}

template <class Derived>
const TraceableTensorShape &
TensorBase<Derived>::dynamic_sizes() const
{
  return _dynamic_sizes;
}

template <class Derived>
TensorShapeRef
TensorBase<Derived>::static_sizes() const
{
  return sizes().slice(dynamic_dim());
}

template <class Derived>
TensorShapeRef
TensorBase<Derived>::intmd_sizes() const
{
  return sizes().slice(dynamic_dim(), intmd_dim());
}

template <class Derived>
TraceableSize
TensorBase<Derived>::batch_size(Size i) const
{
  i = utils::normalize_dim(i, 0, batch_dim());
  if (i < dynamic_dim())
    return _dynamic_sizes[i];
  return size(i);
}

template <class Derived>
Size
TensorBase<Derived>::base_size(Size i) const
{
  i = utils::normalize_dim(i, batch_dim(), dim());
  return size(i);
}

template <class Derived>
const TraceableSize &
TensorBase<Derived>::dynamic_size(Size i) const
{
  i = utils::normalize_dim(i, 0, dynamic_dim());
  return _dynamic_sizes[i];
}

template <class Derived>
Size
TensorBase<Derived>::static_size(Size i) const
{
  i = utils::normalize_dim(i, dynamic_dim(), dim());
  return size(i);
}

template <class Derived>
Size
TensorBase<Derived>::intmd_size(Size i) const
{
  i = utils::normalize_dim(i, dynamic_dim(), batch_dim());
  return size(i);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), static_dim(), indexing::Slice());
  auto res = this->index(indices_vec);
  return Derived(res, res.dim() - static_dim(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices_vec(dynamic_dim(), indexing::Slice());
  indices_vec.insert(indices_vec.end(), indices.begin(), indices.end());
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  auto res = this->index(indices_vec);
  return Derived(res, dynamic_sizes(), res.dim() - dynamic_dim() - base_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_index(indexing::TensorIndicesRef indices) const
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  return neml2::Tensor(this->index(indices2), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_index(indexing::TensorIndicesRef indices) const
{
  neml_assert_dbg(_intmd_dim == 0,
                  "batch_index is only supported when there are no intermediate dimensions.");
  return dynamic_index(indices);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_slice(Size d, const indexing::Slice & index) const
{
  d = utils::normalize_dim(d, 0, dynamic_dim());
  auto res = this->slice(
      d, index.start().expect_int(), index.stop().expect_int(), index.step().expect_int());
  return Derived(res, res.dim() - static_dim(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_slice(Size d, const indexing::Slice & index) const
{
  d = utils::normalize_dim(d, dynamic_dim(), batch_dim());
  auto res = this->slice(
      d, index.start().expect_int(), index.stop().expect_int(), index.step().expect_int());
  return Derived(res, dynamic_sizes(), intmd_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_slice(Size d, const indexing::Slice & index) const
{
  d = utils::normalize_dim(d, batch_dim(), dim());
  auto res = this->slice(
      d, index.start().expect_int(), index.stop().expect_int(), index.step().expect_int());
  return neml2::Tensor(res, dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_slice(Size d, const indexing::Slice & index) const
{
  d = utils::normalize_dim(d, 0, batch_dim());
  if (d < dynamic_dim())
    return dynamic_slice(d, index);
  return intmd_slice(d - dynamic_dim(), index);
}

template <class Derived>
void
TensorBase<Derived>::dynamic_index_put_(indexing::TensorIndicesRef indices, const ATensor & other)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), static_dim(), indexing::Slice());
  this->index_put_(indices_vec, other);
}

template <class Derived>
void
TensorBase<Derived>::dynamic_index_put_(indexing::TensorIndicesRef indices, const CScalar & v)
{
  indexing::TensorIndices indices_vec(indices);
  indices_vec.insert(indices_vec.end(), static_dim(), indexing::Slice());
  this->index_put_(indices_vec, v);
}

template <class Derived>
void
TensorBase<Derived>::intmd_index_put_(indexing::TensorIndicesRef indices, const ATensor & other)
{
  indexing::TensorIndices indices_vec(dynamic_dim(), indexing::Slice());
  indices_vec.insert(indices_vec.end(), indices.begin(), indices.end());
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  this->index_put_(indices_vec, other);
}

template <class Derived>
void
TensorBase<Derived>::intmd_index_put_(indexing::TensorIndicesRef indices, const CScalar & v)
{
  indexing::TensorIndices indices_vec(dynamic_dim(), indexing::Slice());
  indices_vec.insert(indices_vec.end(), indices.begin(), indices.end());
  indices_vec.insert(indices_vec.end(), base_dim(), indexing::Slice());
  this->index_put_(indices_vec, v);
}

template <class Derived>
void
TensorBase<Derived>::base_index_put_(indexing::TensorIndicesRef indices, const ATensor & other)
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  this->index_put_(indices2, other);
}

template <class Derived>
void
TensorBase<Derived>::base_index_put_(indexing::TensorIndicesRef indices, const CScalar & v)
{
  indexing::TensorIndices indices2(batch_dim(), indexing::Slice());
  indices2.insert(indices2.end(), indices.begin(), indices.end());
  this->index_put_(indices2, v);
}

template <class Derived>
void
TensorBase<Derived>::batch_index_put_(indexing::TensorIndicesRef indices, const ATensor & other)
{
  neml_assert_dbg(_intmd_dim == 0,
                  "batch_index_put_ is only supported when there are no intermediate dimensions.");
  dynamic_index_put_(indices, other);
}

template <class Derived>
void
TensorBase<Derived>::batch_index_put_(indexing::TensorIndicesRef indices, const CScalar & v)
{
  neml_assert_dbg(_intmd_dim == 0,
                  "batch_index_put_ is only supported when there are no intermediate dimensions.");
  dynamic_index_put_(indices, v);
}

template <class Derived>
Derived
TensorBase<Derived>::variable_data() const
{
  return Derived(ATensor::variable_data(), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_expand(const TraceableTensorShape & shape) const
{
  // We don't want to touch the other dimensions, so put -1 for them.
  auto net = shape.concrete();
  net.insert(net.end(), static_dim(), -1);

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (std::size_t i = 0; i < shape.size(); ++i)
      if (const auto * const si = shape[i].traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem("size", net.size(), i, *si);

  return Derived(expand(net), shape, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_expand(TensorShapeRef shape) const
{
  if (intmd_sizes() == shape)
    return *this;

  // Unsqueeze missing dimensions
  neml_assert_dbg(Size(shape.size()) >= intmd_dim(),
                  "Invalid intermediate shape to expand. Expected at least ",
                  intmd_dim(),
                  " dimensions.");
  auto tmp = intmd_unsqueeze(0, shape.size() - intmd_dim());

  // We don't want to touch the other dimensions, so put -1 for them.
  TensorShape net(dynamic_dim(), -1);
  net.insert(net.end(), shape.begin(), shape.end());
  net.insert(net.end(), base_dim(), -1);
  return Derived(tmp.expand(net), dynamic_sizes(), Size(shape.size()));
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand(TensorShapeRef shape) const
{
  if (base_sizes() == shape)
    return *this;

  // Unsqueeze missing dimensions
  neml_assert_dbg(Size(shape.size()) >= base_dim(),
                  "Invalid base shape to expand. Expected at least ",
                  base_dim(),
                  " dimensions.");
  auto tmp = base_unsqueeze(0, shape.size() - base_dim());

  // We don't want to touch the batch dimensions, so put -1 for them.
  TensorShape net(batch_dim(), -1);
  net.insert(net.end(), shape.begin(), shape.end());
  return neml2::Tensor(tmp.expand(net), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_expand(const TraceableTensorShape & dynamic_shape,
                                  TensorShapeRef intmd_shape) const
{
  // Unsqueeze missing dimensions
  neml_assert_dbg(Size(intmd_shape.size()) >= intmd_dim(),
                  "Invalid intermediate shape to expand. Expected at least ",
                  intmd_dim(),
                  " dimensions.");
  auto tmp = intmd_unsqueeze(0, intmd_shape.size() - intmd_dim());

  // We don't want to touch the other dimensions, so put -1 for them.
  auto net = utils::add_shapes(dynamic_shape.concrete(), intmd_shape);
  net.insert(net.end(), base_dim(), -1);

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (std::size_t i = 0; i < dynamic_shape.size(); ++i)
      if (const auto * const si = dynamic_shape[i].traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem("size", net.size(), i, *si);

  return Derived(tmp.expand(net), dynamic_shape, tmp.intmd_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::static_expand(TensorShapeRef intmd_shape, TensorShapeRef base_shape) const
{
  auto net = utils::add_shapes(intmd_shape, base_shape);
  if (static_sizes() == net)
    return *this;

  // Unsqueeze missing dimensions
  neml_assert_dbg(Size(intmd_shape.size()) >= intmd_dim(),
                  "Invalid intermediate shape to expand. Expected at least ",
                  intmd_dim(),
                  " dimensions.");
  neml_assert_dbg(Size(base_shape.size()) >= base_dim(),
                  "Invalid base shape to expand. Expected at least ",
                  base_dim(),
                  " dimensions.");
  auto tmp = intmd_unsqueeze(0, intmd_shape.size() - intmd_dim());
  tmp = tmp.base_unsqueeze(0, base_shape.size() - base_dim());

  // We don't want to touch the other dimensions, so put -1 for them.
  net.insert(net.begin(), dynamic_dim(), -1);
  return neml2::Tensor(tmp.expand(net), dynamic_sizes(), tmp.intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_expand(const TraceableSize & size, Size d) const
{
  if (dynamic_size(d) == size)
    return Derived(*this);

  d = utils::normalize_dim(d, 0, dynamic_dim());
  auto shape = dynamic_sizes();
  shape[d] = size;
  return dynamic_expand(shape);
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_expand(Size size, Size d) const
{
  if (intmd_size(d) == size)
    return *this;

  d = utils::normalize_dim(d, dynamic_dim(), batch_dim());
  TensorShape net(dim(), -1);
  net[d] = size;
  return Derived(expand(net), dynamic_sizes(), intmd_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand(Size size, Size d) const
{
  if (base_size(d) == size)
    return *this;

  d = utils::normalize_dim(d, batch_dim(), dim());
  TensorShape net(dim(), -1);
  net[d] = size;
  return neml2::Tensor(expand(net), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_expand_as(const neml2::Tensor & other) const
{
  return dynamic_expand(other.dynamic_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_expand_as(const neml2::Tensor & other) const
{
  return intmd_expand(other.intmd_sizes());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_expand_as(const neml2::Tensor & other) const
{
  return base_expand(other.base_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_expand_as(const neml2::Tensor & other) const
{
  return batch_expand(other.dynamic_sizes(), other.intmd_sizes());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::static_expand_as(const neml2::Tensor & other) const
{
  return static_expand(other.intmd_sizes(), other.base_sizes());
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_reshape(const TraceableTensorShape & shape) const
{
  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (std::size_t i = 0; i < shape.size(); ++i)
      if (const auto * const si = shape[i].traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "shape", shape.size() + static_dim(), i, *si);

  return Derived(reshape(utils::add_shapes(shape.concrete(), static_sizes())), shape, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_reshape(TensorShapeRef shape) const
{
  auto intmd_dim = Size(shape.size());

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (Size i = 0; i < dynamic_dim(); ++i)
      if (const auto * const si = dynamic_size(i).traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "shape", dynamic_dim() + intmd_dim + base_dim(), i, *si);

  return neml2::Tensor(reshape(utils::add_shapes(dynamic_sizes().concrete(), shape, base_sizes())),
                       dynamic_sizes(),
                       intmd_dim);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_reshape(TensorShapeRef shape) const
{
  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (Size i = 0; i < dynamic_dim(); ++i)
      if (const auto * const si = dynamic_size(i).traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "shape", batch_dim() + shape.size(), i, *si);

  return neml2::Tensor(
      reshape(utils::add_shapes(batch_sizes().concrete(), shape)), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_reshape(const TraceableTensorShape & dynamic_shape,
                                   TensorShapeRef intmd_shape) const
{
  auto intmd_dim = Size(intmd_shape.size());

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (std::size_t i = 0; i < dynamic_shape.size(); ++i)
      if (const auto * const si = dynamic_shape[i].traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "shape", dynamic_shape.size() + intmd_dim + base_dim(), i, *si);

  return Derived(reshape(utils::add_shapes(dynamic_shape.concrete(), intmd_shape, base_sizes())),
                 dynamic_shape,
                 intmd_dim);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::static_reshape(TensorShapeRef intmd_shape, TensorShapeRef base_shape) const
{
  auto intmd_dim = Size(intmd_shape.size());

  // Record the dynamic sizes in the traced graph if we are tracing
  if (jit::tracer::isTracing())
    for (Size i = 0; i < dynamic_dim(); ++i)
      if (const auto * const si = dynamic_size(i).traceable())
        jit::tracer::ArgumentStash::stashIntArrayRefElem(
            "shape", dynamic_dim() + intmd_dim + base_shape.size(), i, *si);

  return Derived(reshape(utils::add_shapes(dynamic_sizes().concrete(), intmd_shape, base_shape)),
                 dynamic_sizes(),
                 intmd_dim);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_squeeze(Size d) const
{
  d = utils::normalize_dim(d, 0, dynamic_dim());
  neml_assert(dynamic_size(d) == 1,
              "Cannot squeeze dynamic dimension ",
              d,
              " with size ",
              dynamic_size(d),
              ". Only dimensions of size 1 can be squeezed.");
  auto sizes = dynamic_sizes();
  sizes.erase(sizes.begin() + d); // Remove the squeezed dimension
  return Derived(squeeze(d), sizes, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_squeeze(Size d) const
{
  d = utils::normalize_dim(d, dynamic_dim(), batch_dim());
  return Derived(squeeze(d), dynamic_sizes(), intmd_dim() - 1);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_squeeze(Size d) const
{
  d = utils::normalize_dim(d, batch_dim(), dim());
  return neml2::Tensor(squeeze(d), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_squeeze(Size d) const
{
  d = utils::normalize_dim(d, 0, batch_dim());
  if (d < dynamic_dim())
    return dynamic_squeeze(d);
  return intmd_squeeze(d - dynamic_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_unsqueeze(Size d, Size n) const
{
  neml_assert(n >= 0, "Number of dimensions to unsqueeze must be non-negative.");
  at::Tensor t = *this;
  d = utils::normalize_itr(d, 0, dynamic_dim());
  for (Size i = 0; i < n; ++i)
    t = t.unsqueeze(d);
  auto B = dynamic_sizes();
  B.insert(B.begin() + d, n, 1);
  return Derived(t, B, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_unsqueeze(Size d, Size n) const
{
  neml_assert(n >= 0, "Number of dimensions to unsqueeze must be non-negative.");
  at::Tensor t = *this;
  d = utils::normalize_itr(d, dynamic_dim(), batch_dim());
  for (Size i = 0; i < n; ++i)
    t = t.unsqueeze(d);
  return Derived(t, dynamic_sizes(), intmd_dim() + n);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_unsqueeze(Size d, Size n) const
{
  neml_assert(n >= 0, "Number of dimensions to unsqueeze must be non-negative.");
  at::Tensor t = *this;
  d = utils::normalize_itr(d, batch_dim(), dim());
  for (Size i = 0; i < n; ++i)
    t = t.unsqueeze(d);
  return neml2::Tensor(t, dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_unsqueeze(Size d, Size n) const
{
  d = utils::normalize_itr(d, 0, batch_dim());
  if (d <= dynamic_dim())
    return dynamic_unsqueeze(d, n);
  return intmd_unsqueeze(d - dynamic_dim(), n);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_transpose(Size d1, Size d2) const
{
  d1 = utils::normalize_dim(d1, 0, dynamic_dim());
  d2 = utils::normalize_dim(d2, 0, dynamic_dim());

  auto sizes = dynamic_sizes();
  std::swap(sizes[d1], sizes[d2]);

  return Derived(transpose(d1, d2), sizes, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_transpose(Size d1, Size d2) const
{
  d1 = utils::normalize_dim(d1, dynamic_dim(), batch_dim());
  d2 = utils::normalize_dim(d2, dynamic_dim(), batch_dim());
  return Derived(transpose(d1, d2), dynamic_sizes(), intmd_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_transpose(Size d1, Size d2) const
{
  d1 = utils::normalize_dim(d1, batch_dim(), dim());
  d2 = utils::normalize_dim(d2, batch_dim(), dim());
  return neml2::Tensor(transpose(d1, d2), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_transpose(Size d1, Size d2) const
{
  neml_assert_dbg(_intmd_dim == 0,
                  "batch_transpose is only supported when there are no intermediate dimensions.");
  return dynamic_transpose(d1, d2);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_movedim(Size old_dim, Size new_dim) const
{
  old_dim = utils::normalize_dim(old_dim, 0, dynamic_dim());
  new_dim = utils::normalize_dim(new_dim, 0, dynamic_dim());

  auto sizes = dynamic_sizes();
  auto from = sizes.begin() + old_dim;
  auto to = sizes.begin() + new_dim;
  if (from < to)
    std::rotate(from, from + 1, to + 1);
  else
    std::rotate(to, from, from + 1);

  return Derived(movedim(old_dim, new_dim), sizes, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_movedim(Size old_dim, Size new_dim) const
{
  old_dim = utils::normalize_dim(old_dim, dynamic_dim(), batch_dim());
  new_dim = utils::normalize_dim(new_dim, dynamic_dim(), batch_dim());
  return Derived(movedim(old_dim, new_dim), dynamic_sizes(), intmd_dim());
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_movedim(Size old_dim, Size new_dim) const
{
  old_dim = utils::normalize_dim(old_dim, batch_dim(), dim());
  new_dim = utils::normalize_dim(new_dim, batch_dim(), dim());
  return neml2::Tensor(movedim(old_dim, new_dim), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_movedim(Size old_dim, Size new_dim) const
{
  neml_assert_dbg(_intmd_dim == 0,
                  "batch_movedim is only supported when there are no intermediate dimensions.");
  return dynamic_movedim(old_dim, new_dim);
}

template <class Derived>
Derived
TensorBase<Derived>::dynamic_flatten(Size start_dim, Size end_dim) const
{
  if (dynamic_dim() == 1)
    return *this;

  if (dynamic_dim() == 0)
    return dynamic_unsqueeze(0);

  start_dim = utils::normalize_dim(start_dim, 0, dynamic_dim());
  end_dim = utils::normalize_dim(end_dim, 0, dynamic_dim());
  auto n = utils::traceable_numel(dynamic_sizes().slice(start_dim, end_dim - start_dim + 1));

  return Derived(flatten(start_dim, end_dim), {n}, intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::intmd_flatten(Size start_dim, Size end_dim) const
{
  if (intmd_dim() == 1)
    return *this;

  if (intmd_dim() == 0)
    return intmd_unsqueeze(0);

  start_dim = utils::normalize_dim(start_dim, dynamic_dim(), batch_dim());
  end_dim = utils::normalize_dim(end_dim, dynamic_dim(), batch_dim());

  return Derived(flatten(start_dim, end_dim), dynamic_sizes(), 1);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::base_flatten(Size start_dim, Size end_dim) const
{
  if (base_dim() == 1)
    return *this;

  if (base_dim() == 0)
    return base_unsqueeze(0);

  start_dim = utils::normalize_dim(start_dim, batch_dim(), dim());
  end_dim = utils::normalize_dim(end_dim, batch_dim(), dim());

  return neml2::Tensor(flatten(start_dim, end_dim), dynamic_sizes(), intmd_dim());
}

template <class Derived>
Derived
TensorBase<Derived>::batch_flatten() const
{
  if (intmd_dim() == 0 && dynamic_dim() == 1)
    return *this;

  if (batch_dim() == 0)
    return dynamic_unsqueeze(0);

  auto n = utils::traceable_numel(dynamic_sizes()) * utils::numel(intmd_sizes());
  return Derived(flatten(0, batch_dim() - 1), {n}, 0);
}

template <class Derived>
neml2::Tensor
TensorBase<Derived>::static_flatten() const
{
  if (intmd_dim() == 0 && base_dim() == 1)
    return *this;

  if (static_dim() == 0)
    return base_unsqueeze(0);

  return neml2::Tensor(flatten(dynamic_dim()), dynamic_sizes(), 0);
}

template <class Derived>
Derived
TensorBase<Derived>::operator-() const
{
  return Derived(-ATensor(*this), dynamic_sizes(), intmd_dim());
}

} // end namespace neml2
