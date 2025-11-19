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

#include "neml2/user_tensors/RandomTensor.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <class T>
OptionSet
RandomTensorImpl<T>::expected_options()
{
  OptionSet options = FactoryMethodBase<T>::expected_options();
  options.doc() = "Construct a random " + FactoryMethodBase<T>::tensor_type() +
                  " with given batch shape. Tensor values are set to random values. Values are "
                  "drawn uniformly from the sample space.";

  if constexpr (!std::is_same_v<T, Rot>)
  {
    options.set<double>("min");
    options.set("min").doc() = "Minimum random value.";
    options.set<double>("max");
    options.set("max").doc() = "Maximum random value.";
  }

  return options;
}

template <class T>
RandomTensorImpl<T>::RandomTensorImpl(const OptionSet & options)
  : FactoryMethodBase<T>(options),
    _min(std::is_same_v<T, Rot> ? 0.0 : options.get<double>("min")),
    _max(std::is_same_v<T, Rot> ? 0.0 : options.get<double>("max"))
{
}

template <class T>
T
RandomTensorImpl<T>::make() const
{
  const auto batch_sizes = TensorShapeRef{this->_batch_sizes};
  const auto dynamic_dim = batch_sizes.size() - this->_intmd_dim;
  const auto v0 = T::rand(
      batch_sizes.slice(0, dynamic_dim), batch_sizes.slice(dynamic_dim), default_tensor_options());
  return v0 * (_max - _min) + _min;
}

template <>
Tensor
RandomTensorImpl<Tensor>::make() const
{
  const auto batch_sizes = TensorShapeRef{this->_batch_sizes};
  const auto dynamic_dim = batch_sizes.size() - this->_intmd_dim;
  const auto v0 = Tensor::rand(batch_sizes.slice(0, dynamic_dim),
                               batch_sizes.slice(dynamic_dim),
                               this->_base_sizes,
                               default_tensor_options());
  return v0 * (_max - _min) + _min;
}

template <>
Rot
RandomTensorImpl<Rot>::make() const
{
  const auto batch_sizes = TensorShapeRef{this->_batch_sizes};
  const auto dynamic_dim = batch_sizes.size() - this->_intmd_dim;
  const auto v0 = Rot::rand(
      batch_sizes.slice(0, dynamic_dim), batch_sizes.slice(dynamic_dim), default_tensor_options());
  return v0;
}

#define RANDOMTENSOR_REGISTER(T)                                                                   \
  using Random##T = RandomTensorImpl<T>;                                                           \
  register_NEML2_object_alias(Random##T, "Random" #T)
FOR_ALL_TENSORBASE(RANDOMTENSOR_REGISTER);
} // namespace neml2
