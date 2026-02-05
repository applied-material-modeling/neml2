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

#include "neml2/user_tensors/CenterTensor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/indexing.h"

namespace neml2
{
template <typename T>
OptionSet
CenterTensorTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Compute interval centers along an intermediate dimension.";

  options.set<TensorName<T>>("points");
  options.set("points").doc() = "The input tensor to be centered";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Intermediate dimension to compute centers";

  return options;
}

template <typename T>
CenterTensorTmpl<T>::CenterTensorTmpl(const OptionSet & options)
  : UserTensorBase<T>(options),
    _points(options.get<TensorName<T>>("points")),
    _dim(options.get<Size>("dim"))
{
}

template <typename T>
T
CenterTensorTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f, "Failed assertion: factory != nullptr");

  const auto points = _points.resolve(f);
  const auto n = points.intmd_size(_dim);
  const auto left = points.intmd_slice(_dim, indexing::Slice(0, n - 1));
  const auto right = points.intmd_slice(_dim, indexing::Slice(1, n));
  return 0.5 * (left + right);
}

#define CENTERTENSOR_REGISTER(T)                                                                   \
  using Center##T = CenterTensorTmpl<T>;                                                           \
  register_NEML2_object_alias(Center##T, "Center" #T)
FOR_ALL_TENSORBASE(CENTERTENSOR_REGISTER);
} // namespace neml2
