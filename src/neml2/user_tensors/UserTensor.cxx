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

#include "neml2/user_tensors/UserTensor.h"

#include "neml2/tensors/tensors.h"

namespace neml2
{
template <class T>
OptionSet
UserTensorTmpl<T>::expected_options()
{
  OptionSet options = FactoryMethodBase<T>::expected_options();
  options.doc() =
      "Construct a " + FactoryMethodBase<T>::tensor_type() +
      " from a vector values. The vector will be reshaped according to the specified batch shape.";

  options.set<std::vector<double>>("values");
  options.set("values").doc() = "Values in this (flattened) tensor";

  return options;
}

template <class T>
UserTensorTmpl<T>::UserTensorTmpl(const OptionSet & options)
  : FactoryMethodBase<T>(options),
    _vals(options.get<std::vector<double>>("values"))
{
}

template <class T>
T
UserTensorTmpl<T>::make() const
{
  const auto flat = Tensor::create(_vals, default_tensor_options());

  if (!this->input_options().user_specified("batch_shape") &&
      !this->input_options().user_specified("intermediate_dimension"))
    return flat.base_reshape(this->_base_sizes);

  const auto sizes = utils::add_shapes(this->_batch_sizes, this->_base_sizes);
  return Tensor(
      flat.reshape(sizes), Size(this->_batch_sizes.size()) - this->_intmd_dim, this->_intmd_dim);
}

#define USERTENSOR_REGISTER(T)                                                                     \
  using User##T = UserTensorTmpl<T>;                                                               \
  register_NEML2_object_alias(User##T, #T)
FOR_ALL_TENSORBASE(USERTENSOR_REGISTER);
} // namespace neml2
