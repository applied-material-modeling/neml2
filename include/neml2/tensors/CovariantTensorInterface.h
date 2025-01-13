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
class PrimitiveTensor : public Tensor
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
  /// Clone (take ownership)
  Tensor clone() const;
  /// Discard function graph
  Tensor detach() const;
  /// Change tensor options
  Tensor to(const TensorOptions & options) const;
  /// Negation
  Tensor operator-() const;
  ///@}

  /// @name Getter and setter
  ///@{
  /// Get a tensor by slicing on the batch dimensions
  Tensor batch_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing along a batch dimension
  Tensor batch_slice(Size dim, const indexing::Slice & index) const;
  /// Variable data without function graph
  Tensor variable_data() const;
  ///@}

  /// @name Modifiers
  ///@{
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Tensor batch_expand(const TraceableTensorShape & batch_shape) const;
  /// Return a new view of the tensor with values broadcast along a given batch dimension.
  Tensor batch_expand(const TraceableSize & batch_size, Size dim) const;
  /// Expand the batch to have the same shape as another tensor
  Tensor batch_expand_as(const Tensor & other) const;
  /// Return a new tensor with values broadcast along the batch dimensions.
  Tensor batch_expand_copy(const TraceableTensorShape & batch_shape) const;
  /// Reshape batch dimensions
  Tensor batch_reshape(const TraceableTensorShape & batch_shape) const;
  /// Unsqueeze a batch dimension
  Tensor batch_unsqueeze(Size d) const;
  /// Transpose two batch dimensions
  Tensor batch_transpose(Size d1, Size d2) const;
  ///@}
};
} // namespace neml2
