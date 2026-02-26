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

#include "neml2/user_tensors/InverseSemiInfiniteScalingJacobian.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
template <typename T>
OptionSet
InverseSemiInfiniteScalingJacobianTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Construct a " + UserTensorBase<T>::tensor_type() +
                  " using the inverse semi-infinite scaling Jacobian $(1 - x)^2 / s$.";

  options.set<TensorName<T>>("x");
  options.set("x").doc() = "The input tensor x";

  options.set<TensorName<T>>("s");
  options.set("s").doc() = "The scaling tensor s";

  return options;
}

template <typename T>
InverseSemiInfiniteScalingJacobianTmpl<T>::InverseSemiInfiniteScalingJacobianTmpl(
    const OptionSet & options)
  : UserTensorBase<T>(options),
    _x(options.get<TensorName<T>>("x")),
    _s(options.get<TensorName<T>>("s"))
{
}

template <typename T>
T
InverseSemiInfiniteScalingJacobianTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f, "Failed assertion: factory != nullptr");

  const auto x = _x.resolve(f);
  const auto s = _s.resolve(f);
  const auto one = T::ones_like(x);
  const auto diff = one - x;

  return diff * diff / s;
}

using InverseSemiInfiniteScalingJacobianScalar = InverseSemiInfiniteScalingJacobianTmpl<Scalar>;
register_NEML2_object_alias(InverseSemiInfiniteScalingJacobianScalar,
                            "InverseSemiInfiniteScalingJacobianScalar");

using InverseSemiInfiniteScalingJacobianTensor = InverseSemiInfiniteScalingJacobianTmpl<Tensor>;
register_NEML2_object_alias(InverseSemiInfiniteScalingJacobianTensor,
                            "InverseSemiInfiniteScalingJacobianTensor");
} // namespace neml2
