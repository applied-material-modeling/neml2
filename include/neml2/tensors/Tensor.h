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
#include "neml2/misc/types.h"

namespace neml2
{
// Forward declarations
struct TraceableSize;
struct TraceableTensorShape;

/**
 * Tensor subclasses at::Tensor to additionally keep track of the tensor's batch shape. The first
 * N dimensions of a tensor are considered batch dimensions, and the remaining dimensions are base
 * dimensions. The batch shape is the shape of the batch dimensions, and the base shape is the shape
 * of the base dimensions. The batch shape is stored as a TraceableTensorShape, which is a vector of
 * TraceableSize.
 */
class Tensor
{
public:
  // For developers:
  // When you add a new method to Tensor, you should consider also adding the method to
  // PrimitiveTensor.

  /// Default constructor
  Tensor() = default;

  ///@{
  /// Construct from another at::Tensor with given batch dimension
  Tensor(const at::Tensor & tensor, Size batch_dim);
  Tensor(at::Tensor && tensor, Size batch_dim);
  ///@}

  ///@{
  /// Construct from another at::Tensor with given batch shape
  Tensor(const at::Tensor & tensor, const TraceableTensorShape & batch_shape);
  Tensor(at::Tensor && tensor, TraceableTensorShape && batch_shape);
  ///@}

  /// @{
  /// Special methods
  Tensor(const Tensor & tensor);
  Tensor(Tensor && tensor) noexcept;
  Tensor & operator=(const Tensor & tensor);
  Tensor & operator=(Tensor && tensor) noexcept;
  ~Tensor() = default;
  /// @}

  /// @name Meta operations
  ///@{
  /// Clone (take ownership)
  Tensor clone() const;
  /// Discard function graph
  Tensor detach() const;
  /// Detach from gradient graphs in place
  using at::Tensor::detach_;
  /// Change tensor options
  Tensor to(const TensorOptions & options) const;
  /// Copy another tensor
  using at::Tensor::copy_;
  /// Set all entries to zero
  using at::Tensor::zero_;
  /// Get the requires_grad property
  using at::Tensor::requires_grad;
  /// Set the requires_grad property
  using at::Tensor::requires_grad_;
  ///@}

  ///@{
  /// Operators
  Tensor operator~() const;
  Tensor operator-() const;
  Tensor & operator+=(const Tensor & other);
  Tensor & operator+=(const NScalar & other);
  Tensor & operator-=(const Tensor & other);
  Tensor & operator-=(const NScalar & other);
  Tensor & operator*=(const Tensor & other);
  Tensor & operator*=(const NScalar & other);
  Tensor & operator/=(const Tensor & other);
  Tensor & operator/=(const NScalar & other);
  Tensor & operator&=(const Tensor & other);
  Tensor & operator|=(const Tensor & other);
  Tensor & operator^=(const Tensor & other);
  ///@}

  /// @name Tensor information
  ///@{
  /// Whether this is defined
  using at::Tensor::defined;
  /// Tensor options
  using at::Tensor::options;
  /// Scalar type
  using at::Tensor::scalar_type;
  /// Dtype
  using at::Tensor::dtype;
  /// Device
  using at::Tensor::device;
  /// Number of tensor dimensions
  using at::Tensor::dim;
  /// Tensor shape
  using at::Tensor::sizes;
  /// Size of a dimension
  using at::Tensor::size;
  /// Whether the tensor is batched
  bool batched() const;
  /// Return the number of batch dimensions
  Size batch_dim() const;
  /// Return the number of base dimensions
  Size base_dim() const;
  /// Return the batch size
  const TraceableTensorShape & batch_sizes() const;
  /// Return the size of a batch axis
  TraceableSize batch_size(Size dim) const;
  /// Return the base size
  TensorShapeRef base_sizes() const;
  /// Return the size of a base axis
  Size base_size(Size dim) const;
  /// Return the flattened storage needed just for the base indices
  Size base_storage() const;
  ///@}

  /// @name Getter and setter
  ///@{
  /// Regular tensor indexing
  using at::Tensor::index;
  /// Regular tensor assignment
  using at::Tensor::index_put_;
  /// Get a tensor by slicing on the batch dimensions
  Tensor batch_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing on the base dimensions
  Tensor base_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing along a batch dimension
  Tensor batch_slice(Size dim, const indexing::Slice & index) const;
  /// Get a tensor by slicing along a base dimension
  Tensor base_slice(Size dim, const indexing::Slice & index) const;
  ///@{
  /// Set values by slicing on the batch dimensions
  void batch_index_put_(indexing::TensorIndicesRef indices, const at::Tensor & other);
  void batch_index_put_(indexing::TensorIndicesRef indices, const NScalar & v);
  ///@}
  ///@{
  /// Set values by slicing on the base dimensions
  void base_index_put_(indexing::TensorIndicesRef indices, const at::Tensor & other);
  void base_index_put_(indexing::TensorIndicesRef indices, const NScalar & v);
  ///@}
  /// Variable data without function graph
  Tensor variable_data() const;
  ///@}

  /// @name Modifiers
  ///@{
  /// Plain expand without distinguishing batch and base dimensions
  using at::Tensor::expand;
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Tensor batch_expand(const TraceableTensorShape & batch_shape) const;
  /// Return a new view of the tensor with values broadcast along a given batch dimension.
  Tensor batch_expand(const TraceableSize & batch_size, Size dim) const;
  /// Return a new view of the tensor with values broadcast along the base dimensions.
  Tensor base_expand(TensorShapeRef base_shape) const;
  /// Return a new view of the tensor with values broadcast along a given base dimension.
  Tensor base_expand(Size base_size, Size dim) const;
  /// Plain expand_as without distinguishing batch and base dimensions
  using at::Tensor::expand_as;
  /// Expand the batch to have the same shape as another tensor
  Tensor batch_expand_as(const Tensor & other) const;
  /// Expand the base to have the same shape as another tensor
  Tensor base_expand_as(const Tensor & other) const;
  /// Return a new tensor with values broadcast along the batch dimensions.
  Tensor batch_expand_copy(const TraceableTensorShape & batch_shape) const;
  /// Return a new tensor with values broadcast along the base dimensions.
  Tensor base_expand_copy(TensorShapeRef base_shape) const;
  /// Plain reshape without distinguishing batch and base dimensions
  using at::Tensor::reshape;
  /// Reshape batch dimensions
  Tensor batch_reshape(const TraceableTensorShape & batch_shape) const;
  /// Reshape base dimensions
  Tensor base_reshape(TensorShapeRef base_shape) const;
  /// Plain unsqueeze without distinguishing batch and base dimensions
  using at::Tensor::unsqueeze;
  /// Unsqueeze a batch dimension
  Tensor batch_unsqueeze(Size d) const;
  /// Unsqueeze a base dimension
  Tensor base_unsqueeze(Size d) const;
  /// Plain transpose without distinguishing batch and base dimensions
  using at::Tensor::transpose;
  /// Transpose two batch dimensions
  Tensor batch_transpose(Size d1, Size d2) const;
  /// Transpose two base dimensions
  Tensor base_transpose(Size d1, Size d2) const;
  /// Plain flatten without distinguishing batch and base dimensions
  using at::Tensor::flatten;
  /// Flatten base dimensions
  Tensor base_flatten() const;
  ///@}

private:
  TraceableTensorShape _batch_sizes;
};

Tensor operator+(const Tensor & a, const NScalar & b);
Tensor operator+(const NScalar & a, const Tensor & b);
Tensor operator+(const Tensor & a, const Tensor & b);

Tensor operator-(const Tensor & a, const NScalar & b);
Tensor operator-(const NScalar & a, const Tensor & b);
Tensor operator-(const Tensor & a, const Tensor & b);

Tensor operator*(const Tensor & a, const NScalar & b);
Tensor operator*(const NScalar & a, const Tensor & b);
Tensor operator*(const Tensor & a, const Tensor & b);

Tensor operator/(const Tensor & a, const NScalar & b);
Tensor operator/(const NScalar & a, const Tensor & b);
Tensor operator/(const Tensor & a, const Tensor & b);

namespace math
{
/**
 * @brief Batched matrix-matrix product
 *
 * The input matrices \p a and \p b must have exactly 2 base dimensions. The batch shapes must
 * broadcast.
 */
Tensor bmm(const Tensor & a, const Tensor & b);

/**
 * @brief Batched matrix-vector product
 *
 * The input tensor \p a must have exactly 2 base dimensions.
 * The input tensor \p v must have exactly 1 base dimension.
 * The batch shapes must broadcast.
 */
Tensor bmv(const Tensor & a, const Tensor & v);

/**
 * @brief Batched vector-vector (dot) product
 *
 * The input tensor \p a must have exactly 1 base dimension.
 * The input tensor \p b must have exactly 1 base dimension.
 * The batch shapes must broadcast.
 */
Tensor bvv(const Tensor & a, const Tensor & b);
}
} // namespace neml2
