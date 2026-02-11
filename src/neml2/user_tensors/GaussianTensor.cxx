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

#include "neml2/user_tensors/GaussianTensor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
template <typename T>
OptionSet
GaussianTensorTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Construct a " + UserTensorBase<T>::tensor_type() +
                  " with values sampled from a Gaussian profile at given points. The Gaussian is "
                  "\\f$g(x) = h \\exp(-\\tfrac{1}{2} z^2)\\f$ with \\f$z = (x - c) / w\\f$, where "
                  "\\f$h\\f$ is the height, \\f$w\\f$ is the width, and \\f$c\\f$ is the center.";

  options.set<TensorName<T>>("points");
  options.set("points").doc() = "The coordinates to evaluate the Gaussian at";

  options.set<TensorName<Scalar>>("width");
  options.set("width").doc() = "The Gaussian width";

  options.set<TensorName<Scalar>>("height");
  options.set("height").doc() = "The Gaussian height";

  options.set<TensorName<Scalar>>("center") = TensorName<Scalar>("0");
  options.set("center").doc() = "The Gaussian center in the domain";

  return options;
}

template <typename T>
GaussianTensorTmpl<T>::GaussianTensorTmpl(const OptionSet & options)
  : UserTensorBase<T>(options),
    _points(options.get<TensorName<T>>("points")),
    _width(options.get<TensorName<Scalar>>("width")),
    _height(options.get<TensorName<Scalar>>("height")),
    _center(options.get<TensorName<Scalar>>("center"))
{
}

template <typename T>
T
GaussianTensorTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f, "Failed assertion: factory != nullptr");

  const auto points = _points.resolve(f);
  const auto width = _width.resolve(f);
  const auto height = _height.resolve(f);
  const auto center = _center.resolve(f);

  const auto z = (points - center) / width;
  return height * neml2::exp(-0.5 * z * z);
}

using GaussianScalar = GaussianTensorTmpl<Scalar>;
register_NEML2_object_alias(GaussianScalar, "GaussianScalar");

using GaussianTensor = GaussianTensorTmpl<Tensor>;
register_NEML2_object_alias(GaussianTensor, "GaussianTensor");
} // namespace neml2
