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

#include "neml2/user_tensors/FullTensor.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <class T>
OptionSet
FullTensorTmpl<T>::expected_options()
{
  OptionSet options = FactoryMethodBase<T>::expected_options();
  options.doc() = "Construct a full " + FactoryMethodBase<T>::tensor_type() +
                  " with given batch shape. Tensor values are set to the specified value.";
  options.set<double>("value");
  options.set("value").doc() = "Value to fill the tensor with";
  return options;
}

template <class T>
FullTensorTmpl<T>::FullTensorTmpl(const OptionSet & options)
  : FactoryMethodBase<T>(options),
    _value(options.get<double>("value"))
{
}

template <class T>
T
FullTensorTmpl<T>::make() const
{
  const auto batch_sizes = TensorShapeRef{this->_batch_sizes};
  const auto dynamic_dim = batch_sizes.size() - this->_intmd_dim;
  return Tensor::full(batch_sizes.slice(0, dynamic_dim),
                      batch_sizes.slice(dynamic_dim),
                      this->_base_sizes,
                      _value,
                      default_tensor_options());
}

#define FULLTENSOR_REGISTER(T)                                                                     \
  using Full##T = FullTensorTmpl<T>;                                                               \
  register_NEML2_object_alias(Full##T, "Full" #T)
FOR_ALL_TENSORBASE(FULLTENSOR_REGISTER);
} // namespace neml2
