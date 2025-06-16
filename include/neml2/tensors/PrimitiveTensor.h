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

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>

#include "neml2/misc/errors.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{
using TensorDataContainer = torch::detail::TensorDataContainer;

/**
 * @brief PrimitiveTensor inherits from TensorBase and additionally templates on the base shape.
 *
 * @tparam S Base shape
 */
template <class Derived, Size... S>
class PrimitiveTensor : public TensorBase<Derived>
{
public:
  /// The base shape
  static inline const TensorShape const_base_sizes = {S...};

  /// The base dim
  static constexpr Size const_base_dim = sizeof...(S);

  /// The base storage
  static inline const Size const_base_storage = utils::storage_size({S...});

  /// Special member functions
  PrimitiveTensor() = default;

  /// Construct from another ATensor given batch dimension
  PrimitiveTensor(const ATensor & tensor, Size batch_dim);

  /// Construct from another ATensor given batch shape
  PrimitiveTensor(const ATensor & tensor, const TraceableTensorShape & batch_shape);

  /// Copy constructor
  PrimitiveTensor(const Tensor & tensor);

  /// Construct from another ATensor and infer batch dimension
  explicit PrimitiveTensor(const ATensor & tensor);

  /// Implicit conversion to a Tensor and loses information on the fixed base shape
  operator Tensor() const;

  /// Arbitrary tensor from a nested container with inferred batch dimension
  [[nodiscard]] static Derived create(const TensorDataContainer & data,
                                      const TensorOptions & options = default_tensor_options());
  /// Unbatched empty tensor
  [[nodiscard]] static Derived empty(const TensorOptions & options = default_tensor_options());
  /// Empty tensor given batch shape
  [[nodiscard]] static Derived empty(const TraceableTensorShape & batch_shape,
                                     const TensorOptions & options = default_tensor_options());
  /// Unbatched zero tensor
  [[nodiscard]] static Derived zeros(const TensorOptions & options = default_tensor_options());
  /// Zero tensor given batch shape
  [[nodiscard]] static Derived zeros(const TraceableTensorShape & batch_shape,
                                     const TensorOptions & options = default_tensor_options());
  /// Unbatched unit tensor
  [[nodiscard]] static Derived ones(const TensorOptions & options = default_tensor_options());
  /// Unit tensor given batch shape
  [[nodiscard]] static Derived ones(const TraceableTensorShape & batch_shape,
                                    const TensorOptions & options = default_tensor_options());
  /// Unbatched tensor filled with a given value given base shape
  [[nodiscard]] static Derived full(const CScalar & init,
                                    const TensorOptions & options = default_tensor_options());
  /// Full tensor given batch shape
  [[nodiscard]] static Derived full(const TraceableTensorShape & batch_shape,
                                    const CScalar & init,
                                    const TensorOptions & options = default_tensor_options());

  /// Derived tensor classes should define identity_map where appropriate
  [[nodiscard]] static Tensor identity_map(const TensorOptions &)
  {
    throw NEMLException("Not implemented");
  }
};

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor, Size batch_dim)
  : TensorBase<Derived>(tensor, batch_dim)
{
#ifndef NDEBUG
  if (this->base_sizes() != const_base_sizes)
    throw NEMLException("Base shape mismatch");
#endif
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor,
                                                const TraceableTensorShape & batch_shape)
  : TensorBase<Derived>(tensor, batch_shape)
{
#ifndef NDEBUG
  if (this->base_sizes() != const_base_sizes)
    throw NEMLException("Base shape mismatch");
#endif
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const Tensor & tensor)
  : TensorBase<Derived>(tensor)
{
#ifndef NDEBUG
  if (this->base_sizes() != const_base_sizes)
    throw NEMLException("Base shape mismatch");
#endif
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor)
  : TensorBase<Derived>(tensor, tensor.dim() - const_base_dim)
{
#ifndef NDEBUG
  if (this->base_sizes() != const_base_sizes)
    throw NEMLException("Base shape mismatch");
#endif
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::operator Tensor() const
{
  return Tensor(*this, this->batch_sizes());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::create(const TensorDataContainer & data,
                                       const TensorOptions & options)
{
  return Derived(torch::autograd::make_variable(
      data.convert_to_tensor(options.requires_grad(false)), options.requires_grad()));
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(const TensorOptions & options)
{
  return Tensor::empty(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(const TraceableTensorShape & batch_shape,
                                      const TensorOptions & options)
{
  return Tensor::empty(batch_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(const TensorOptions & options)
{
  return Tensor::zeros(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(const TraceableTensorShape & batch_shape,
                                      const TensorOptions & options)
{
  return Tensor::zeros(batch_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(const TensorOptions & options)
{
  return Tensor::ones(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(const TraceableTensorShape & batch_shape,
                                     const TensorOptions & options)
{
  return Tensor::ones(batch_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(const CScalar & init, const TensorOptions & options)
{
  return Tensor::full(const_base_sizes, init, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(const TraceableTensorShape & batch_shape,
                                     const CScalar & init,
                                     const TensorOptions & options)
{
  return Tensor::full(batch_shape, const_base_sizes, init, options);
}
} // namespace neml2
