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

#include <ATen/core/Tensor.h>

#include "neml2/jit/TraceableTensorShape.h"
#include "neml2/tensors/functions/operators.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
// Forward declarations
template <class Derived>
class TensorBase;

class Tensor;

/**
 * @brief NEML2's enhanced tensor type.
 *
 * neml2::TensorBase derives from ATensor and clearly distinguishes between "batched"
 * dimensions from other dimensions. The shape of the "batched" dimensions is called the batch size,
 * and the shape of the rest dimensions is called the base size.
 */
template <class Derived>
class TensorBase : public ATensor
{
public:
  /// Special member functions
  TensorBase() = default;

  /// Construct from another ATensor with given batch dimension
  TensorBase(const ATensor & tensor, Size batch_dim);

  /// Construct from another ATensor with given batch shape
  TensorBase(const ATensor & tensor, const TraceableTensorShape & batch_shape);

  /// Copy constructor
  TensorBase(const neml2::Tensor & tensor);

  TensorBase(Real) = delete;

  /// \addtogroup tensor_creation Tensor creation API
  ///@{
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
  ///@}

  /// @name Meta operations
  ///@{
  /// Clone (take ownership)
  Derived clone() const;
  /// Discard function graph
  Derived detach() const;
  /// Detach from gradient graphs in place
  using ATensor::detach_;
  /// Change tensor options
  Derived to(const TensorOptions & options) const;
  /// Copy another tensor
  using ATensor::copy_;
  /// Set all entries to zero
  using ATensor::zero_;
  /// Get the requires_grad property
  using ATensor::requires_grad;
  /// Set the requires_grad property
  using ATensor::requires_grad_;
  /// Negation
  Derived operator-() const;
  ///@}

  /// @name Tensor information
  ///@{
  /// Tensor options
  using ATensor::options;
  /// Scalar type
  using ATensor::scalar_type;
  /// Device
  using ATensor::device;
  /// Number of tensor dimensions
  using ATensor::dim;
  /// Tensor shape
  using ATensor::sizes;
  /// Size of a dimension
  using ATensor::size;
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
  using ATensor::index;
  using ATensor::index_put_;
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
  void batch_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
  void batch_index_put_(indexing::TensorIndicesRef indices, Real v);
  ///@}
  ///@{
  /// Set values by slicing on the base dimensions
  void base_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
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
  Derived batch_expand_as(const neml2::Tensor & other) const;
  /// Expand the base to have the same shape as another tensor
  neml2::Tensor base_expand_as(const neml2::Tensor & other) const;
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

protected:
  /// Number of batch dimensions. The first `_batch_dim` dimensions are considered batch dimensions.
  Size _batch_dim = {};

  /// Traceable batch sizes
  TraceableTensorShape _batch_sizes;
};
} // namespace neml2
