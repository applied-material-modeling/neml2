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

#include "neml2/tensors/TraceableSize.h"
#include "neml2/tensors/TraceableTensorShape.h"
#include "neml2/tensors/functions/operators.h"
#include "neml2/tensors/functions/logical.h"
#include "neml2/tensors/indexing.h"
#include "neml2/tensors/macros.h"

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
 * dimensions from other dimensions.
 *
 * A tensor shape can be pictorially labeled as below
 *
 * ```
 * (b1, b2, b3, b4, b5, b6, b7, ...; d1, d2, d3, d4, d5, ...)
 *  |_____________________________|  |_____________________|
 *               batch                        base
 *  |____________|  |_____________|
 *      dynamic      intermediate
 *                  |______________________________________|
 *                                  static
 * ```
 *
 * - *batch*: The dimensions over which tensor operations should broadcast, i.e., the same operation
 *            should be applied across all batches.
 * - *base*: The dimensions that are statically (at compile-time) determined by the tensor type.
 * - *dynamic*: The dimensions whose sizes are dynamic in a traced graph of tensor operations,
 *              i.e., the traced graph can be generalized to tensors with different dynamic sizes.
 * - *static*: The dimensions whose sizes are fixed in a traced graph of tensor operations,
 *             i.e., the sizes of these dimensions are hard-coded in the traced graph.
 * - *intermediate*: Batch dimensions that are static, i.e., their sizes are fixed in the traced
 *                   graph but tensor operations are batched over them.
 *
 * @note By default, and in most cases, the number of intermediate dimensions is zero, meaning that
 * "batch" == "dynamic" and "base" == "static".
 */
template <class Derived>
class TensorBase : public ATensor
{
public:
  /// Default constructor
  TensorBase() = default;

  /// Construct from an ATensor with given dynamic dimension
  TensorBase(const ATensor & tensor, Size dynamic_dim, Size intmd_dim);

  /// Construct from an ATensor with given dynamic shape
  TensorBase(const ATensor & tensor, TraceableTensorShape dynamic_shape, Size intmd_dim);

  /// Copy constructor
  template <class Derived2>
  TensorBase(const TensorBase<Derived2> & tensor)
    : TensorBase(tensor, tensor.dynamic_sizes(), tensor.intmd_dim())
  {
  }

  TensorBase(double) = delete;
  TensorBase(float) = delete;
  TensorBase(int) = delete;

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
  [[nodiscard]] static Derived full_like(const Derived & other, const CScalar & init);
  /// Random tensor like another, i.e. same batch and base shapes, same tensor options, etc.
  /// The random values follow a uniform distribution [0, 1)
  [[nodiscard]] static Derived rand_like(const Derived & other);
  ///@}

  /// @name Meta operations
  ///@{
  /// Make contiguous
  Derived contiguous() const;
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
  /// Defined
  using ATensor::defined;
  /// Tensor options
  using ATensor::options;
  /// Scalar type
  using ATensor::scalar_type;
  /// Device
  using ATensor::device;
  ///@}

  /// @return the number of dimensions
  ///@{
  using ATensor::dim;
  Size batch_dim() const;
  Size base_dim() const;
  Size dynamic_dim() const;
  Size static_dim() const;
  Size intmd_dim() const;
  ///@}

  /// @return the tensor shape
  ///@{
  using ATensor::sizes;
  TraceableTensorShape batch_sizes() const;
  TensorShapeRef base_sizes() const;
  const TraceableTensorShape & dynamic_sizes() const;
  TensorShapeRef static_sizes() const;
  TensorShapeRef intmd_sizes() const;
  ///@}

  /// @return the size of dimension @p i
  ///@{
  using ATensor::size;
  TraceableSize batch_size(Size i) const;
  Size base_size(Size i) const;
  const TraceableSize & dynamic_size(Size i) const;
  Size static_size(Size i) const;
  Size intmd_size(Size i) const;
  ///@}

  /// Regular tensor indexing
  using ATensor::index;
  using ATensor::index_put_;

  ///@{
  /// Get a tensor by slicing along multiple dimensions
  Derived dynamic_index(indexing::TensorIndicesRef indices) const;
  Derived intmd_index(indexing::TensorIndicesRef indices) const;
  neml2::Tensor base_index(indexing::TensorIndicesRef indices) const;
  Derived batch_index(indexing::TensorIndicesRef indices) const;
  ///@}

  ///@{
  /// Get a tensor by slicing along a dimension
  Derived dynamic_slice(Size d, const indexing::Slice & index) const;
  Derived intmd_slice(Size d, const indexing::Slice & index) const;
  neml2::Tensor base_slice(Size d, const indexing::Slice & index) const;
  Derived batch_slice(Size d, const indexing::Slice & index) const;
  ///@}

  ///@{
  /// Set values by slicing along multiple dimensions
  void dynamic_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
  void dynamic_index_put_(indexing::TensorIndicesRef indices, const CScalar & v);
  void intmd_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
  void intmd_index_put_(indexing::TensorIndicesRef indices, const CScalar & v);
  void base_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
  void base_index_put_(indexing::TensorIndicesRef indices, const CScalar & v);
  void batch_index_put_(indexing::TensorIndicesRef indices, const ATensor & other);
  void batch_index_put_(indexing::TensorIndicesRef indices, const CScalar & v);
  ///@}

  /// Variable data without function graph
  Derived variable_data() const;

  ///@{
  /// Return a view of the tensor with values broadcast along multiple dimensions.
  Derived dynamic_expand(const TraceableTensorShape & shape) const;
  Derived intmd_expand(TensorShapeRef shape) const;
  neml2::Tensor base_expand(TensorShapeRef shape) const;
  Derived batch_expand(const TraceableTensorShape & dynamic_shape,
                       TensorShapeRef intmd_shape = {}) const;
  neml2::Tensor static_expand(TensorShapeRef intmd_shape, TensorShapeRef base_shape) const;
  ///@}

  ///@{
  /// Return a view of the tensor with values broadcast along the given dimension.
  Derived dynamic_expand(const TraceableSize & size, Size d) const;
  Derived intmd_expand(Size size, Size d) const;
  neml2::Tensor base_expand(Size size, Size d) const;
  ///@}

  ///@{
  /// Expand the dimensions to have the same shape as another tensor's
  Derived dynamic_expand_as(const neml2::Tensor & other) const;
  Derived intmd_expand_as(const neml2::Tensor & other) const;
  neml2::Tensor base_expand_as(const neml2::Tensor & other) const;
  Derived batch_expand_as(const neml2::Tensor & other) const;
  neml2::Tensor static_expand_as(const neml2::Tensor & other) const;
  ///@}

  ///@{
  /// Reshape the dimension group
  Derived dynamic_reshape(const TraceableTensorShape & shape) const;
  Derived intmd_reshape(TensorShapeRef shape) const;
  neml2::Tensor base_reshape(TensorShapeRef shape) const;
  Derived batch_reshape(const TraceableTensorShape & dynamic_shape,
                        TensorShapeRef intmd_shape = {}) const;
  neml2::Tensor static_reshape(TensorShapeRef intmd_shape, TensorShapeRef base_shape) const;
  ///@}

  ///@{
  /// Squeeze a dimension
  Derived dynamic_squeeze(Size d) const;
  Derived intmd_squeeze(Size d) const;
  neml2::Tensor base_squeeze(Size d) const;
  Derived batch_squeeze(Size d) const;
  ///@}

  ///@{
  /// Unsqueeze @p n dimensions at @p d
  Derived dynamic_unsqueeze(Size d, Size n = 1) const;
  Derived intmd_unsqueeze(Size d, Size n = 1) const;
  neml2::Tensor base_unsqueeze(Size d, Size n = 1) const;
  Derived batch_unsqueeze(Size d, Size n = 1) const;
  ///@}

  ///@{
  /// Transpose two dimensions
  Derived dynamic_transpose(Size d1, Size d2) const;
  Derived intmd_transpose(Size d1, Size d2) const;
  neml2::Tensor base_transpose(Size d1, Size d2) const;
  Derived batch_transpose(Size d1, Size d2) const;
  ///@}

  ///@{}
  /// Move a dimension to a new position
  Derived dynamic_movedim(Size old_dim, Size new_dim) const;
  Derived intmd_movedim(Size old_dim, Size new_dim) const;
  neml2::Tensor base_movedim(Size old_dim, Size new_dim) const;
  Derived batch_movedim(Size old_dim, Size new_dim) const;
  ///@}

  ///@{
  /// Flatten a dimension group
  Derived dynamic_flatten(Size start_dim = 0, Size end_dim = -1) const;
  Derived intmd_flatten(Size start_dim = 0, Size end_dim = -1) const;
  neml2::Tensor base_flatten(Size start_dim = 0, Size end_dim = -1) const;
  /**
   * @brief Flatten batch dimensions
   *
   * @note All intermediate dimensions and dynamic dimensions get flattened into one single batch
   * dimension. In other words, the resulting tensor will have ZERO intermediate dimensions.
   */
  Derived batch_flatten() const;
  /**
   * @brief Flatten static dimensions
   *
   * @note All intermediate dimensions and base dimensions get flattened into one single base
   * dimension. In other words, the resulting tensor will have ZERO intermediate dimensions.
   */
  neml2::Tensor static_flatten() const;
  ///@}

protected:
  /// Validate shapes and dimensions
  void validate_shapes_and_dims() const;

private:
  /// Sizes of dynamic dimensions
  TraceableTensorShape _dynamic_sizes;

  /// Number of intermediate dimensions
  // BTW, "intmd" is an abbreviation for "intermediate" found in the Merriam-Webster dictionary :)
  Size _intmd_dim = 0;
};

// Export TensorBase so other TU don't repeat the instantiation
#define EXPORT_TENSORBASE(T) extern template class TensorBase<T>
FOR_ALL_TENSORBASE(EXPORT_TENSORBASE);
#undef EXPORT_TENSORBASE
} // namespace neml2
