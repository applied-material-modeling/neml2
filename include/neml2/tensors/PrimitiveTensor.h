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

#include <ATen/ScalarOps.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <tuple>
#include <type_traits>

#include "neml2/misc/defaults.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/functions/utils.h"
#include "neml2/tensors/functions/stack.h"

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
  /// Base shape sequence
  using base_sizes_sequence = std::integer_sequence<Size, S...>;

  /// The base shape
  static inline const TensorShape const_base_sizes = {S...};

  /// The base dim
  static constexpr Size const_base_dim = sizeof...(S);

  /// The base numel
  static constexpr Size const_base_numel = (1 * ... * S);

  /// Special member functions
  PrimitiveTensor() = default;

  /// Construct from an ATensor and infer dynamic shape
  PrimitiveTensor(const ATensor & tensor, Size intmd_dim);

  /// Construct from an ATensor and extract dynamic shape given dynamic dimension
  PrimitiveTensor(const ATensor & tensor, Size dynamic_dim, Size intmd_dim);

  /// Construct from an ATensor given dynamic shape
  PrimitiveTensor(const ATensor & tensor,
                  const TraceableTensorShape & dynamic_shape,
                  Size intmd_dim);

  /// Copy constructor
  template <class Derived2>
  PrimitiveTensor(const TensorBase<Derived2> & tensor);

  /// Implicit conversion to a Tensor (discards information on the fixed base shape)
  operator neml2::Tensor() const;

  /// Arbitrary tensor from a nested container with inferred batch dimension
  [[nodiscard]] static Derived create(const TensorDataContainer & data,
                                      Size intmd_dim = 0,
                                      const TensorOptions & options = default_tensor_options());

  /// Empty tensor with undefined values
  ///@{
  [[nodiscard]] static Derived empty(const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived empty(const TraceableTensorShape & dynamic_shape,
                                     TensorShapeRef intmd_shape = {},
                                     const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with zeros
  ///@{
  [[nodiscard]] static Derived zeros(const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived zeros(const TraceableTensorShape & dynamic_shape,
                                     TensorShapeRef intmd_shape = {},
                                     const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with ones
  ///@{
  [[nodiscard]] static Derived ones(const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived ones(const TraceableTensorShape & dynamic_shape,
                                    TensorShapeRef intmd_shape = {},
                                    const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with a given value
  ///@{
  [[nodiscard]] static Derived full(const CScalar & init,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived full(const TraceableTensorShape & dynamic_shape,
                                    TensorShapeRef intmd_shape,
                                    const CScalar & init,
                                    const TensorOptions & options = default_tensor_options());
  ///@}

  /// Tensor filled with random values from a uniform distribution [0, 1)
  ///@{
  [[nodiscard]] static Derived rand(const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived rand(const TraceableTensorShape & dynamic_shape,
                                    TensorShapeRef intmd_shape,
                                    const TensorOptions & options = default_tensor_options());
  ///@}

  /// Fill tensor with values
  ///@{
  template <typename... Args,
            typename = std::enable_if_t<(sizeof...(Args) == const_base_numel ||
                                         sizeof...(Args) == const_base_numel + 1)>>
  [[nodiscard]] static Derived fill(Args &&... args);
  ///@}

  /// Einstein summation along base dimensions
  [[nodiscard]] static Derived einsum(c10::string_view equation, TensorList tensors);

  /// Single-element accessor
  template <typename... Args>
  Scalar operator()(Args... i) const;

protected:
  /// Validate shapes and dimensions
  void validate_shapes_and_dims() const;
};

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor, Size intmd_dim)
  : TensorBase<Derived>(tensor, tensor.dim() - const_base_dim - intmd_dim, intmd_dim)
{
  validate_shapes_and_dims();
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor,
                                                Size dynamic_dim,
                                                Size intmd_dim)
  : TensorBase<Derived>(tensor, dynamic_dim, intmd_dim)
{
  if (dynamic_dim + intmd_dim + const_base_dim != tensor.dim())
    throw NEMLException("Inconsistent dimensions when constructing PrimitiveTensor. Expected "
                        "tensor to have dynamic dimension " +
                        std::to_string(dynamic_dim) + ", intmd dimension " +
                        std::to_string(intmd_dim) + ", and base dimension " +
                        std::to_string(const_base_dim) + ", but tensor has " +
                        std::to_string(tensor.dim()) + " dimensions.");
  validate_shapes_and_dims();
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const ATensor & tensor,
                                                const TraceableTensorShape & dynamic_shape,
                                                Size intmd_dim)
  : TensorBase<Derived>(tensor, dynamic_shape, intmd_dim)
{
  validate_shapes_and_dims();
}

template <class Derived, Size... S>
template <class Derived2>
PrimitiveTensor<Derived, S...>::PrimitiveTensor(const TensorBase<Derived2> & tensor)
  : TensorBase<Derived>(tensor)
{
  validate_shapes_and_dims();
}

template <class Derived, Size... S>
void
PrimitiveTensor<Derived, S...>::validate_shapes_and_dims() const
{
#ifndef NDEBUG
  if (this->base_sizes() != const_base_sizes)
    throw NEMLException("Base shape mismatch");
#endif
}

template <class Derived, Size... S>
PrimitiveTensor<Derived, S...>::
operator neml2::Tensor() const
{
  return neml2::Tensor(*this, this->dynamic_sizes(), this->intmd_dim());
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::create(const TensorDataContainer & data,
                                       Size intmd_dim,
                                       const TensorOptions & options)
{
  return Derived(torch::autograd::make_variable(
                     data.convert_to_tensor(options.requires_grad(false)), options.requires_grad()),
                 intmd_dim)
      .clone(); // clone to take ownership of the data
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(const TensorOptions & options)
{
  return Tensor::empty(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::empty(const TraceableTensorShape & dynamic_shape,
                                      TensorShapeRef intmd_shape,
                                      const TensorOptions & options)
{
  return Tensor::empty(dynamic_shape, intmd_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(const TensorOptions & options)
{
  return Tensor::zeros(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::zeros(const TraceableTensorShape & dynamic_shape,
                                      TensorShapeRef intmd_shape,
                                      const TensorOptions & options)
{
  return Tensor::zeros(dynamic_shape, intmd_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(const TensorOptions & options)
{
  return Tensor::ones(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::ones(const TraceableTensorShape & dynamic_shape,
                                     TensorShapeRef intmd_shape,
                                     const TensorOptions & options)
{
  return Tensor::ones(dynamic_shape, intmd_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(const CScalar & init, const TensorOptions & options)
{
  return Tensor::full(const_base_sizes, init, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::full(const TraceableTensorShape & dynamic_shape,
                                     TensorShapeRef intmd_shape,
                                     const CScalar & init,
                                     const TensorOptions & options)
{
  return Tensor::full(dynamic_shape, intmd_shape, const_base_sizes, init, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::rand(const TensorOptions & options)
{
  return Tensor::rand(const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::rand(const TraceableTensorShape & dynamic_shape,
                                     TensorShapeRef intmd_shape,
                                     const TensorOptions & options)
{
  return Tensor::rand(dynamic_shape, intmd_shape, const_base_sizes, options);
}

template <class Derived, Size... S>
Derived
PrimitiveTensor<Derived, S...>::einsum(c10::string_view equation, TensorList tensors)
{
  const auto [tensors_aligned, i] = utils::align_intmd_dim(tensors);
  std::vector<ATensor> tensors_einsum(tensors_aligned.size());
  for (std::size_t j = 0; j < tensors_aligned.size(); ++j)
    tensors_einsum[j] = tensors_aligned[j];
  auto res = at::einsum(equation, tensors_einsum);
  return Derived(res, i);
}

template <class Tuple, std::size_t... I>
auto
make_tensors(Tuple && t, std::index_sequence<I...>, const TensorOptions & options)
{
  return std::vector<neml2::Tensor>{
      neml2::Tensor(at::scalar_to_tensor(std::get<I>(std::forward<Tuple>(t)), options.device())
                        .to(options.dtype()),
                    0)...};
}

template <class Derived, Size... S>
template <typename... Args, typename>
Derived
PrimitiveTensor<Derived, S...>::fill(Args &&... args)
{
  if constexpr (sizeof...(Args) == const_base_numel)
  {
    if constexpr ((std::is_convertible_v<Args, neml2::Tensor> && ...))
    {
#ifndef NDEBUG
      neml_assert_dbg(((args.base_dim() == 0) && ...),
                      "All input tensors must be scalar-like (no base dimensions)");
#endif
      return base_stack({std::forward<Args>(args)...}).base_reshape(const_base_sizes);
    }
    else if constexpr ((std::is_convertible_v<Args, CScalar> && ...))
    {
      auto t = neml2::Tensor::create({std::forward<Args>(args)...}, default_tensor_options());
      return t.base_reshape(const_base_sizes);
    }
  }
  else if constexpr (sizeof...(Args) == const_base_numel + 1)
  {
    auto tup = std::forward_as_tuple(std::forward<Args>(args)...);
    const auto & options = std::get<sizeof...(Args) - 1>(tup);
    auto vals = make_tensors(tup, std::make_index_sequence<sizeof...(Args) - 1>{}, options);
    return base_stack(vals).base_reshape(const_base_sizes);
  }

  throw NEMLException("Invalid argument types to PrimitiveTensor::fill");
}

} // namespace neml2
