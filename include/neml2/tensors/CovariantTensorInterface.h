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

#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * PrimitiveTensor is a mixin class for tensors with base shape known at compile time.
 *
 * It provides return type wrapping for tensor operations utilizing the Curiously Recurring Template
 * Pattern (CRTP).
 */
template <class Derived, Size... S>
class PrimitiveTensor : protected Tensor
{
public:
  /// The base shape
  static inline const TensorShape const_base_sizes = {S...};

  /// The base dim
  static constexpr Size const_base_dim = sizeof...(S);

  /// The base storage
  static inline const Size const_base_storage = utils::storage_size({S...});

  /// Default constructor
  PrimitiveTensor() = default;

  /// Construct from another torch::Tensor and infer batch dimension
  PrimitiveTensor(const torch::Tensor & tensor);

  /// Construct from another Tensor
  PrimitiveTensor(const Tensor & tensor);

  /// @name Meta operations
  ///@{
  /// Change tensor options
  Derived to(const TensorOptions & options) const;
  /// Clone (take ownership)
  Derived clone() const;
  /// Discard function graph
  Derived detach() const;
  /// Detach from gradient graphs in place
  Derived & detach_();
  /// Copy another tensor
  Derived & copy_(const at::Tensor & other);
  /// Set all entries to zero
  Derived & zero_();
  /// Get the requires_grad property
  using Tensor::requires_grad;
  /// Set the requires_grad property
  Derived & requires_grad_(bool req);
  ///@}

  /// @name Tensor information
  ///@{
  /// Whether this is defined
  using Tensor::defined;
  /// Tensor options
  using Tensor::options;
  /// Scalar type
  using Tensor::scalar_type;
  /// Dtype
  using Tensor::dtype;
  /// Device
  using Tensor::device;
  /// Number of tensor dimensions
  using Tensor::dim;
  /// Tensor shape
  using Tensor::sizes;
  /// Size of a dimension
  using Tensor::size;
  /// Whether the tensor is batched
  using Tensor::batched;
  /// Return the number of batch dimensions
  using Tensor::batch_dim;
  /// Return the number of base dimensions
  using Tensor::base_dim;
  /// Return the batch size
  using Tensor::batch_sizes;
  /// Return the size of a batch axis
  using Tensor::batch_size;
  /// Return the base size
  using Tensor::base_sizes;
  /// Return the size of a base axis
  Size base_size(Size dim) const;
  /// Return the flattened storage needed just for the base indices
  Size base_storage() const;
  ///@}

  /// @name Getter and setter
  ///@{
  /// Regular tensor indexing
  using Tensor::index;
  /// Regular tensor assignment
  using Tensor::index_put_;
  /// Get a tensor by slicing on the batch dimensions
  Derived batch_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing on the base dimensions
  using Tensor::base_index;
  /// Get a tensor by slicing along a batch dimension
  Derived batch_slice(Size dim, const indexing::Slice & index) const;
  /// Get a tensor by slicing along a base dimension
  using Tensor::base_slice;
  ///@{
  /// Set values by slicing on the batch dimensions
  Derived & batch_index_put_(indexing::TensorIndicesRef indices, const at::Tensor & other);
  Derived & batch_index_put_(indexing::TensorIndicesRef indices, const LScalar & v);
  ///@}
  ///@{
  /// Set values by slicing on the base dimensions
  using Tensor::base_index_put_;
  ///@}
  /// Variable data without function graph
  Derived variable_data() const;
  ///@}

  /// @name Modifiers
  ///@{
  /// Plain expand without distinguishing batch and base dimensions
  using Tensor::expand;
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Derived batch_expand(const TraceableTensorShape & batch_shape) const;
  /// Return a new view of the tensor with values broadcast along a given batch dimension.
  Derived batch_expand(const TraceableSize & batch_size, Size dim) const;
  /// Return a new view of the tensor with values broadcast along the base dimensions.
  /// Return a new view of the tensor with values broadcast along a given base dimension.
  using Tensor::base_expand;
  /// Plain expand_as without distinguishing batch and base dimensions
  using Tensor::expand_as;
  /// Expand the batch to have the same shape as another tensor
  Derived batch_expand_as(const Tensor & other) const;
  /// Expand the base to have the same shape as another tensor
  using Tensor::base_expand_as;
  /// Return a new tensor with values broadcast along the batch dimensions.
  Derived batch_expand_copy(const TraceableTensorShape & batch_shape) const;
  /// Return a new tensor with values broadcast along the base dimensions.
  using Tensor::base_expand_copy;
  /// Plain reshape without distinguishing batch and base dimensions
  using Tensor::reshape;
  /// Reshape batch dimensions
  Derived batch_reshape(const TraceableTensorShape & batch_shape) const;
  /// Reshape base dimensions
  using Tensor::base_reshape;
  /// Plain unsqueeze without distinguishing batch and base dimensions
  using Tensor::unsqueeze;
  /// Unsqueeze a batch dimension
  Derived batch_unsqueeze(Size d) const;
  /// Unsqueeze a base dimension
  using Tensor::base_unsqueeze;
  /// Plain transpose without distinguishing batch and base dimensions
  using Tensor::transpose;
  /// Transpose two batch dimensions
  Derived batch_transpose(Size d1, Size d2) const;
  /// Transpose two base dimensions
  using Tensor::base_transpose;
  /// Plain flatten without distinguishing batch and base dimensions
  using Tensor::flatten;
  /// Flatten base dimensions
  using Tensor::base_flatten;
  ///@}

  ///@{
  /// Operators
  Derived operator~() const;
  Derived operator-() const;
  Derived & operator+=(const Tensor & other);
  Derived & operator+=(const LScalar & other);
  Derived & operator-=(const Tensor & other);
  Derived & operator-=(const LScalar & other);
  Derived & operator*=(const Tensor & other);
  Derived & operator*=(const LScalar & other);
  Derived & operator/=(const Tensor & other);
  Derived & operator/=(const LScalar & other);
  ///@}
};
} // namespace neml2
