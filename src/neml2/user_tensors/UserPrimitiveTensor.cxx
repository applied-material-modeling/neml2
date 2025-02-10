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

#include "neml2/user_tensors/UserPrimitiveTensor.h"

#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <typename T>
OptionSet
UserPrimitiveTensor<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = UserTensorBase::expected_options();
  options.doc() =
      "Construct a " + tensor_type +
      " from a vector values. The vector will be reshaped according to the specified batch shape.";

  options.set<std::vector<Real>>("values");
  options.set("values").doc() = "Values in this (flattened) tensor";

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  return options;
}

template <typename T>
UserPrimitiveTensor<T>::UserPrimitiveTensor(const OptionSet & options)
  : T(T::empty(options.get<TensorShape>("batch_shape"), default_tensor_options())),
    UserTensorBase(options)
{
  auto vals = options.get<std::vector<Real>>("values");
  auto flat = Tensor::create(vals, default_tensor_options());
  if (vals.size() == size_t(this->base_storage()))
    this->index_put_({indexing::Ellipsis}, flat.reshape(this->base_sizes()));
  else if (vals.size() == size_t(utils::storage_size(this->sizes())))
    this->index_put_({indexing::Ellipsis}, flat.reshape(this->sizes()));
  else
    neml_assert(false,
                "Number of values ",
                vals.size(),
                " must equal to either the base storage size ",
                this->base_storage(),
                " or the total storage size ",
                utils::storage_size(this->sizes()));
}

#define USERPRIMITIVETENSOR_REGISTER(T)                                                            \
  using User##T = UserPrimitiveTensor<T>;                                                          \
  register_NEML2_object_alias(User##T, #T)
FOR_ALL_PRIMITIVETENSOR(USERPRIMITIVETENSOR_REGISTER);
} // namespace neml2
