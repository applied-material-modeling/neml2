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

#include "neml2/user_tensors/DifferenceTensor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/diff.h"

namespace neml2
{
template <typename T>
OptionSet
DifferenceTensorTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Compute finite differences along an intermediate dimension.";

  options.set<TensorName<T>>("points");
  options.set("points").doc() = "The input tensor to be differenced";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Intermediate dimension to take the finite difference";

  return options;
}

template <typename T>
DifferenceTensorTmpl<T>::DifferenceTensorTmpl(const OptionSet & options)
  : UserTensorBase<T>(options),
    _points(options.get<TensorName<T>>("points")),
    _dim(options.get<Size>("dim"))
{
}

template <typename T>
T
DifferenceTensorTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f, "Failed assertion: factory != nullptr");

  return intmd_diff(_points.resolve(f), 1, _dim);
}

#define DIFFERENCETENSOR_REGISTER(T)                                                               \
  using Difference##T = DifferenceTensorTmpl<T>;                                                   \
  register_NEML2_object_alias(Difference##T, "Difference" #T)
FOR_ALL_TENSORBASE(DIFFERENCETENSOR_REGISTER);
} // namespace neml2
