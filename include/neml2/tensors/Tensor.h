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

#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>

#include "neml2/misc/types.h"
#include "neml2/tensors/TensorBase.h"
#include "neml2/misc/defaults.h"

namespace neml2
{
class Tensor;
using TensorDataContainer = torch::detail::TensorDataContainer;

namespace utils
{
/// @brief Find the broadcast batch shape of all the tensors
/// The returned batch shape will be _traceable_. @see neml2::TraceableTensorShape
TraceableTensorShape broadcast_batch_sizes(const std::vector<Tensor> & tensors);
} // namespace utils

class Tensor : public TensorBase<Tensor>
{
public:
  /// Special member functions
  Tensor() = default;

  /// Construct from another ATensor
  Tensor(const ATensor & tensor, Size batch_dim);

  /// Construct from another ATensor with given batch shape
  Tensor(const ATensor & tensor, const TraceableTensorShape & batch_shape);

  /// Copy from TensorBase
  template <class Derived>
  Tensor(const TensorBase<Derived> & tensor)
    : TensorBase(tensor, tensor.batch_sizes())
  {
  }

  /// Arbitrary (unbatched) tensor from a nested container
  [[nodiscard]] static Tensor create(const TensorDataContainer & data,
                                     const TensorOptions & options = default_tensor_options());
  /// Arbitrary tensor from a nested container
  [[nodiscard]] static Tensor create(const TensorDataContainer & data,
                                     Size batch_dim,
                                     const TensorOptions & options = default_tensor_options());
  /// Unbatched empty tensor given base shape
  [[nodiscard]] static Tensor empty(TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  /// Empty tensor given batch and base shapes
  [[nodiscard]] static Tensor empty(const TraceableTensorShape & batch_shape,
                                    TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  /// Unbatched tensor filled with zeros given base shape
  [[nodiscard]] static Tensor zeros(TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  /// Zero tensor given batch and base shapes
  [[nodiscard]] static Tensor zeros(const TraceableTensorShape & batch_shape,
                                    TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  /// Unbatched tensor filled with ones given base shape
  [[nodiscard]] static Tensor ones(TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  /// Unit tensor given batch and base shapes
  [[nodiscard]] static Tensor ones(const TraceableTensorShape & batch_shape,
                                   TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  /// Unbatched tensor filled with a given value given base shape
  [[nodiscard]] static Tensor full(TensorShapeRef base_shape,
                                   Real init,
                                   const TensorOptions & options = default_tensor_options());
  /// Full tensor given batch and base shapes
  [[nodiscard]] static Tensor full(const TraceableTensorShape & batch_shape,
                                   TensorShapeRef base_shape,
                                   Real init,
                                   const TensorOptions & options = default_tensor_options());
  /// Unbatched identity tensor
  [[nodiscard]] static Tensor identity(Size n,
                                       const TensorOptions & options = default_tensor_options());
  /// Identity tensor given batch shape and base length
  [[nodiscard]] static Tensor identity(const TraceableTensorShape & batch_shape,
                                       Size n,
                                       const TensorOptions & options = default_tensor_options());

  /// Expand base dimension to a given size
  Tensor base_unsqueeze_to(Size n) const;
};
} // namespace neml2
