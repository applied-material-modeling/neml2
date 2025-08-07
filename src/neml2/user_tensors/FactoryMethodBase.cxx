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

#include "neml2/user_tensors/FactoryMethodBase.h"
#include "neml2/misc/types.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
template <class T>
OptionSet
FactoryMethodBase<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();

  options.set<TensorShape>("batch_shape") = {};
  options.set("batch_shape").doc() = "Batch shape";

  options.set<unsigned int>("intermediate_dimension") = 0;
  options.set("intermediate_dimension").doc() = "Intermediate dimension";

  options.set<TensorShape>("base_shape") = {};
  options.set("base_shape").doc() = "Base shape";

  if constexpr (!std::is_same_v<T, Tensor>)
  {
    options.set<TensorShape>("base_shape") = T::const_base_sizes;
    options.set("base_shape").suppressed() = true;
  }

  return options;
}

template <class T>
FactoryMethodBase<T>::FactoryMethodBase(const OptionSet & options)
  : UserTensorBase<T>(options),
    _batch_sizes(options.get<TensorShape>("batch_shape")),
    _base_sizes(options.get<TensorShape>("base_shape")),
    _intmd_dim(options.get<unsigned int>("intermediate_dimension"))
{
  neml_assert(_intmd_dim <= _batch_sizes.size(),
              "Intermediate dimension ",
              _intmd_dim,
              " must be less than or equal to the number of batch dimensions ",
              _batch_sizes.size());
}

#define INSTANTIATE_FACTORYMETHODBASE(T) template class FactoryMethodBase<T>
FOR_ALL_TENSORBASE(INSTANTIATE_FACTORYMETHODBASE);
} // namespace neml2
