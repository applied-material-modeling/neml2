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
using TensorList = c10::ArrayRef<neml2::Tensor>;

namespace utils
{
/// @brief Find the broadcast dynamic shape of all the tensors
/// The returned dynamic shape will be _traceable_. @see neml2::TraceableTensorShape
TraceableTensorShape broadcast_dynamic_sizes(const std::vector<Tensor> & tensors);
/// @brief Find the broadcast intermediate shape of all the tensors
TensorShape broadcast_intmd_sizes(const std::vector<Tensor> & tensors);
} // namespace utils

class Tensor : public TensorBase<Tensor>
{
public:
  /// Special member functions
  Tensor() = default;

  /// Construct from another ATensor with inferred dynamic shape
  Tensor(const ATensor & tensor, Size dynamic_dim, Size intmd_dim = 0);

  /// Construct from another ATensor with given dynamic shape
  Tensor(const ATensor & tensor, const TraceableTensorShape & dynamic_shape, Size intmd_dim = 0);

  /// Copy from TensorBase
  template <class Derived>
  Tensor(const TensorBase<Derived> & tensor)
    : TensorBase(tensor, tensor.dynamic_sizes(), tensor.intmd_dim())
  {
  }

  /// Arbitrary (unbatched) tensor from a nested container
  [[nodiscard]] static Tensor create(const TensorDataContainer & data,
                                     const TensorOptions & options = default_tensor_options());

  /// Arbitrary tensor from a nested container
  [[nodiscard]] static Tensor create(const TensorDataContainer & data,
                                     Size dynamic_dim,
                                     Size intmd_dim = 0,
                                     const TensorOptions & options = default_tensor_options());

  /// Empty tensor filled with undefined values
  ///@{
  [[nodiscard]] static Tensor empty(TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Tensor empty(const TraceableTensorShape & dynamic_shape,
                                    TensorShapeRef intmd_shape,
                                    TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with zeros
  ///@{
  [[nodiscard]] static Tensor zeros(TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Tensor zeros(const TraceableTensorShape & dynamic_shape,
                                    TensorShapeRef intmd_shape,
                                    TensorShapeRef base_shape,
                                    const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with ones
  ///@{
  [[nodiscard]] static Tensor ones(TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Tensor ones(const TraceableTensorShape & dynamic_shape,
                                   TensorShapeRef intmd_shape,
                                   TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with a given value
  ///@{
  [[nodiscard]] static Tensor full(TensorShapeRef base_shape,
                                   const CScalar & init,
                                   const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Tensor full(const TraceableTensorShape & dynamic_shape,
                                   TensorShapeRef intmd_shape,
                                   TensorShapeRef base_shape,
                                   const CScalar & init,
                                   const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with random values from a uniform distribution [0, 1)
  ///@{
  [[nodiscard]] static Tensor rand(TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Tensor rand(const TraceableTensorShape & dynamic_shape,
                                   TensorShapeRef intmd_shape,
                                   TensorShapeRef base_shape,
                                   const TensorOptions & options = default_tensor_options());
  ///@}

  /// Identity tensor
  [[nodiscard]] static Tensor identity(Size n,
                                       const TensorOptions & options = default_tensor_options());
};
} // namespace neml2
