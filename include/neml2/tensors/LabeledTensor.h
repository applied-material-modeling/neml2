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

#include "neml2/misc/types.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
/**
 * @brief The primary data structure in NEML2 for working with labeled tensor views.
 *
 * Each LabeledTensor consists of one Tensor and one or more LabeledAxis. The
 * `LabeledTensor<D>` is templated on the base dimension \f$D\f$. LabeledTensor handles the
 * creation, modification, and accessing of labeled tensors.
 *
 * @tparam D The number of base dimensions
 */
template <class Derived, Size D>
class LabeledTensor
{
public:
  /// Default constructor
  LabeledTensor() = default;

  /// Construct from a torch::Tensor and array of `LabeledAxis`
  LabeledTensor(const torch::Tensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Construct from a Tensor with array of `LabeledAxis`
  LabeledTensor(const Tensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Copy constructor
  LabeledTensor(const Derived & other);

  /// Assignment operator
  LabeledTensor<Derived, D> & operator=(const Derived & other);

  ///@{
  /// Implicit conversion
  operator Tensor() const;
  operator torch::Tensor() const;
  ///@}

  /// Setup new empty storage
  [[nodiscard]] static Derived
  empty(TensorShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new storage with zeros
  [[nodiscard]] static Derived
  zeros(TensorShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  ///@{
  /// Get the underlying tensor
  const Tensor & tensor() const { return _tensor; }
  Tensor & tensor() { return _tensor; }
  ///@}

  /// @name Meta operations
  // These methods mirror TensorBase
  ///@{
  /// Clone this LabeledTensor
  Derived clone(torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous) const;
  /// Return a copy without gradient graphs
  Derived detach() const;
  /// Detach from gradient graphs
  void detach_();
  /// Change tensor options
  Derived to(const torch::TensorOptions & options) const;
  /// Copy another tensor
  void copy_(const torch::Tensor & other);
  /// Set all entries to zero
  void zero_();
  /// Get the requires_grad property
  bool requires_grad() const;
  /// Set the requires_grad property
  void requires_grad_(bool req = true);
  /// Negation
  Derived operator-() const;
  ///@}

  /// @name Tensor information
  // These methods mirror TensorBase
  ///@{
  /// Tensor options
  torch::TensorOptions options() const;
  /// Tensor options
  torch::Dtype scalar_type() const;
  /// Tensor options
  torch::Device device() const;
  /// Number of tensor dimensions
  Size dim() const;
  /// Tensor shape
  TensorShapeRef sizes() const;
  /// Tensor shape
  Size size(Size dim) const;
  /// Whether the tensor is batched
  bool batched() const;
  /// Return the number of batch dimensions
  Size batch_dim() const;
  /// Return the number of base dimensions
  constexpr Size base_dim() const { return D; }
  /// Return the batch size
  TensorShapeRef batch_sizes() const;
  /// Return the length of some batch axis
  Size batch_size(Size d) const;
  /// Return the base size
  TensorShapeRef base_sizes() const;
  /// Return the length of some base axis
  Size base_size(Size d) const;
  /// Return the flattened storage needed just for the base indices
  Size base_storage() const;
  ///@}

  /// @name Getter and setter
  // These methods mirror TensorBase
  ///@{
  /// Get a tensor by slicing on the batch dimensions
  Derived batch_index(indexing::TensorIndicesRef indices) const;
  /// Get a tensor by slicing on the base dimensions
  Tensor base_index(indexing::TensorLabelsRef labels) const;
  /// Set values by slicing on the batch dimensions
  void batch_index_put_(indexing::TensorIndicesRef indices, const torch::Tensor & other);
  /// Set values by slicing on the base dimensions
  void base_index_put_(indexing::TensorLabelsRef labels, const Tensor & other);
  ///@}

  /// Get a tensor by slicing on the base dimensions AND reinterpret it as a primitive tensor
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<TensorBase<T>, T>>>
  T reinterpret(indexing::TensorLabelsRef indices) const;

  /// Get all the labeled axes
  const std::array<const LabeledAxis *, D> & axes() const { return _axes; }
  /// Get a specific labeled axis
  const LabeledAxis & axis(Size i = 0) const { return *_axes[i]; }

  /**
   * @brief Fill with another LabeledTensor that matches this one on all but one axis
   *
   * @param Derived The LabeledTensor to fill from
   * @param Size The common axis, default 0
   * @param bool If true, fill recursively down subaxes
   */
  void fill(const Derived & other, Size odim = 0, bool recursive = true);

protected:
  /// The tensor
  Tensor _tensor;

  /// The labeled axes of this tensor
  std::array<const LabeledAxis *, D> _axes;

private:
  /// Get the storage shape for the slices
  TensorShape storage_shape(indexing::TensorLabelsRef) const;

  /// Get slicing indices given the names on each axis
  indexing::TensorIndices labels_to_indices(indexing::TensorLabelsRef) const;
};

template <class Derived, Size D>
template <typename T, typename>
T
LabeledTensor<Derived, D>::reinterpret(indexing::TensorLabelsRef indices) const
{
  return base_index(indices).base_reshape(T::const_base_sizes);
}
} // namespace neml2
