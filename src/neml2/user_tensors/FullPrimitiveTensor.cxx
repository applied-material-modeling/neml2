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

#include "neml2/user_tensors/FullPrimitiveTensor.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
OptionSet
FullPrimitiveTensor<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = UserTensorBase::expected_options();
  options.doc() =
      "Construct a full " + tensor_type + " with given batch shape filled with a given value.";

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<Real>("value");
  options.set("value").doc() = "Value used to fill the tensor";

  return options;
}

template <typename T>
FullPrimitiveTensor<T>::FullPrimitiveTensor(const OptionSet & options)
  : T(T::full(options.get<TensorShape>("batch_shape"),
              options.get<Real>("value"),
              default_tensor_options())),
    UserTensorBase(options)
{
}

#define FULLPRIMITIVETENSOR_REGISTER(T)                                                            \
  using Full##T = FullPrimitiveTensor<T>;                                                          \
  register_NEML2_object_alias(Full##T, "Full" #T)
FOR_ALL_PRIMITIVETENSOR(FULLPRIMITIVETENSOR_REGISTER);
} // namespace neml2
