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

#include "neml2/misc/utils.h"

namespace neml2
{
// Forward declarations
template <class Derived>
class TensorBase;

class Tensor;

/**
 * @brief NEML2's enhanced tensor type.
 *
 * neml2::TensorBase derives from torch::Tensor and clearly distinguishes between "batched"
 * dimensions from other dimensions. The shape of the "batched" dimensions is called the batch size,
 * and the shape of the rest dimensions is called the base size.
 */
template <class Derived>
class TensorBase : public torch::Tensor
{
public:
  /// Default constructor
  TensorBase() = default;

  /// Construct from another torch::Tensor with given batch dimension
  TensorBase(const torch::Tensor & tensor, Size batch_dim);

  /// Construct from another torch::Tensor with given batch shape
  TensorBase(const torch::Tensor & tensor, const TraceableTensorShape & batch_shape);

  /// Copy constructor
  template <class Derived2>
  TensorBase(const TensorBase<Derived2> & tensor);

  TensorBase(Real) = delete;

  /// Empty tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived empty_like(const Derived & other);
  /// Zero tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived zeros_like(const Derived & other);
  /// Unit tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  [[nodiscard]] static Derived ones_like(const Derived & other);
  /// Full tensor like another, i.e. same batch and base shapes, same tensor options, etc.,
  /// but filled with a different value
  [[nodiscard]] static Derived full_like(const Derived & other, Real init);

  /**
   * @brief Create a new tensor by adding a new batch dimension with linear spacing between \p
   * start and \p end.
   *
   * \p start and \p end must be broadcastable. The new batch dimension will be added at the
   * user-specified dimension \p dim which defaults to 0.
   *
   * For example, if \p start has shape `(3, 2; 5, 5)` and \p end has shape `(3, 1; 5, 5)`, then
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cpp
   * linspace(start, end, 100, 1);
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   * will have shape `(3, 100, 2; 5, 5)`, note the location of the new dimension and the
   * broadcasting.
   *
   * @param start The starting tensor
   * @param end The ending tensor
   * @param nstep The number of steps with even spacing along the new dimension
   * @param dim Where to insert the new dimension
   * @return Tensor Linearly spaced tensor
   */
  [[nodiscard]] static Derived
  linspace(const Derived & start, const Derived & end, Size nstep, Size dim = 0);
  /// log-space equivalent of the linspace named constructor
  [[nodiscard]] static Derived
  logspace(const Derived & start, const Derived & end, Size nstep, Size dim = 0, Real base = 10);

  /// @name Meta operations
  ///@{
  /// Clone (take ownership)
  Derived clone(torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous) const;
  /// Discard function graph
  Derived detach() const;
  /// Detach from gradient graphs in place
  using torch::Tensor::detach_;
  /// Change tensor options
  Derived to(const torch::TensorOptions & options) const;
  /// Copy another tensor
  using torch::Tensor::copy_;
  /// Set all entries to zero
  using torch::Tensor::zero_;
  /// Get the requires_grad property
  using torch::Tensor::requires_grad;
  /// Set the requires_grad property
  using torch::Tensor::requires_grad_;
  /// Negation
  Derived operator-() const;
  ///@}

  /// @name Tensor information
  ///@{
  /// Tensor options
  using torch::Tensor::options;
  /// Scalar type
  using torch::Tensor::scalar_type;
  /// Device
  using torch::Tensor::device;
  /// Number of tensor dimensions
  using torch::Tensor::dim;
  /// Tensor shape
  using torch::Tensor::sizes;
  /// Size of a dimension
  using torch::Tensor::size;
  /// Whether the tensor is batched
  bool batched() const;
  /// Return the number of batch dimensions
  Size batch_dim() const;
  /// Return the number of base dimensions
  Size base_dim() const;
  /// Return the batch size
  const TraceableTensorShape & batch_sizes() const;
  /// Return the size of a batch axis
  TraceableSize batch_size(Size index) const;
  /// Return the base size
  TensorShapeRef base_sizes() const;
  /// Return the size of a base axis
  Size base_size(Size index) const;
  /// Return the flattened storage needed just for the base indices
  Size base_storage() const;
  ///@}

  /// @name Getter and setter
  ///@{
  /// Regular tensor indexing
  using torch::Tensor::index;
  using torch::Tensor::index_put_;
  /// Get a tensor by slicing on the batch dimensions
  Derived batch_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing on the base dimensions
  neml2::Tensor base_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing along a batch dimension
  Derived batch_slice(Size dim, const indexing::Slice & index) const;
  /// Get a tensor by slicing along a base dimension
  neml2::Tensor base_slice(Size dim, const indexing::Slice & index) const;
  ///@{
  /// Set values by slicing on the batch dimensions
  void batch_index_put_(indexing::TensorIndicesRef indices, const torch::Tensor & other);
  void batch_index_put_(indexing::TensorIndicesRef indices, Real v);
  ///@}
  ///@{
  /// Set values by slicing on the base dimensions
  void base_index_put_(indexing::TensorIndicesRef indices, const torch::Tensor & other);
  void base_index_put_(indexing::TensorIndicesRef indices, Real v);
  ///@}
  /// Variable data without function graph
  Derived variable_data() const;
  ///@}

  /// @name Modifiers
  ///@{
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Derived batch_expand(const TraceableTensorShape & batch_shape) const;
  /// Return a new view of the tensor with values broadcast along a given batch dimension.
  Derived batch_expand(const TraceableSize & batch_size, Size dim) const;
  /// Return a new view of the tensor with values broadcast along the base dimensions.
  neml2::Tensor base_expand(TensorShapeRef base_shape) const;
  /// Return a new view of the tensor with values broadcast along a given base dimension.
  neml2::Tensor base_expand(Size base_size, Size dim) const;
  /// Expand the batch to have the same shape as another tensor
  template <class Derived2>
  Derived batch_expand_as(const Derived2 & other) const;
  /// Expand the base to have the same shape as another tensor
  template <class Derived2>
  Derived2 base_expand_as(const Derived2 & other) const;
  /// Return a new tensor with values broadcast along the batch dimensions.
  Derived batch_expand_copy(const TraceableTensorShape & batch_shape) const;
  /// Return a new tensor with values broadcast along the base dimensions.
  neml2::Tensor base_expand_copy(TensorShapeRef base_shape) const;
  /// Reshape batch dimensions
  Derived batch_reshape(const TraceableTensorShape & batch_shape) const;
  /// Reshape base dimensions
  neml2::Tensor base_reshape(TensorShapeRef base_shape) const;
  /// Unsqueeze a batch dimension
  Derived batch_unsqueeze(Size d) const;
  /// Unsqueeze a base dimension
  neml2::Tensor base_unsqueeze(Size d) const;
  /// Transpose two batch dimensions
  Derived batch_transpose(Size d1, Size d2) const;
  /// Transpose two base dimensions
  neml2::Tensor base_transpose(Size d1, Size d2) const;
  /// Flatten base dimensions
  neml2::Tensor base_flatten() const;
  ///@}

private:
  /// Number of batch dimensions. The first `_batch_dim` dimensions are considered batch dimensions.
  Size _batch_dim = {};

  /// Traceable batch sizes
  TraceableTensorShape _batch_sizes;
};

template <class Derived>
template <class Derived2>
TensorBase<Derived>::TensorBase(const TensorBase<Derived2> & tensor)
  : _batch_dim(tensor.batch_dim()),
    _batch_sizes(tensor.batch_sizes())
{
  torch::Tensor::operator=(tensor);
}

template <class Derived>
template <class Derived2>
Derived
TensorBase<Derived>::batch_expand_as(const Derived2 & other) const
{
  return batch_expand(other.batch_sizes());
}

template <class Derived>
template <class Derived2>
Derived2
TensorBase<Derived>::base_expand_as(const Derived2 & other) const
{
  return base_expand(other.base_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator+(const Derived & a, const Real & b)
{
  return Derived(torch::operator+(a, b), a.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator+(const Real & a, const Derived & b)
{
  return b + a;
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator+(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator+(a, b), broadcast_batch_dim(a, b));
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator-(const Derived & a, const Real & b)
{
  return Derived(torch::operator-(a, b), a.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator-(const Real & a, const Derived & b)
{
  return -b + a;
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator-(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator-(a, b), broadcast_batch_dim(a, b));
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator*(const Derived & a, const Real & b)
{
  return Derived(torch::operator*(a, b), a.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator*(const Real & a, const Derived & b)
{
  return b * a;
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator/(const Derived & a, const Real & b)
{
  return Derived(torch::operator/(a, b), a.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator/(const Real & a, const Derived & b)
{
  return Derived(torch::operator/(a, b), b.batch_sizes());
}

template <class Derived,
          typename = typename std::enable_if_t<std::is_base_of_v<TensorBase<Derived>, Derived>>>
Derived
operator/(const Derived & a, const Derived & b)
{
  neml_assert_broadcastable_dbg(a, b);
  return Derived(torch::operator/(a, b), broadcast_batch_dim(a, b));
}
} // namespace neml2
