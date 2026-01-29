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
#include "neml2/tensors/functions/linspace.h"

namespace neml2
{
template <typename T>
OptionSet
GaussianTensorTmpl<T>::expected_options()
{
  OptionSet options = UserTensorBase<T>::expected_options();
  options.doc() = "Construct a " + UserTensorBase<T>::tensor_type() +
                  " with values sampled from a Gaussian profile over a linspace domain.";

  options.set<TensorName<T>>("start");
  options.set("start").doc() = "The starting coordinate of the domain";

  options.set<TensorName<T>>("end");
  options.set("end").doc() = "The ending coordinate of the domain";

  options.set<TensorName<Scalar>>("width");
  options.set("width").doc() = "The Gaussian width";

  options.set<TensorName<Scalar>>("height");
  options.set("height").doc() = "The Gaussian height";

  options.set<TensorName<Scalar>>("center") = TensorName<Scalar>("0");
  options.set("center").doc() = "The Gaussian center in the domain";

  options.set<Size>("nstep");
  options.set("nstep").doc() = "The number of points in the domain";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Where to insert the new dimension";

  EnumSelection selection({"dynamic", "intermediate"}, "dynamic");
  options.set<EnumSelection>("group") = selection;
  options.set("group").doc() =
      "Dimension group to apply the operation. Options are: " + selection.join();

  return options;
}

template <typename T>
GaussianTensorTmpl<T>::GaussianTensorTmpl(const OptionSet & options)
  : UserTensorBase<T>(options),
    _start(options.get<TensorName<T>>("start")),
    _end(options.get<TensorName<T>>("end")),
    _width(options.get<TensorName<Scalar>>("width")),
    _height(options.get<TensorName<Scalar>>("height")),
    _center(options.get<TensorName<Scalar>>("center")),
    _nstep(options.get<Size>("nstep")),
    _dim(options.get<Size>("dim")),
    _group(options.get<EnumSelection>("group"))
{
}

template <typename T>
T
GaussianTensorTmpl<T>::make() const
{
  auto * f = this->factory();
  neml_assert(f, "Failed assertion: factory != nullptr");

  const auto start = _start.resolve(f);
  const auto end = _end.resolve(f);
  const auto width = _width.resolve(f);
  const auto height = _height.resolve(f);
  const auto center = _center.resolve(f);

  T x = (_group == "dynamic") ? dynamic_linspace(start, end, _nstep, _dim)
                              : intmd_linspace(start, end, _nstep, _dim);

  const auto z = (x - center) / width;
  return height * neml2::exp(-0.5 * z * z);
}

using GaussianScalar = GaussianTensorTmpl<Scalar>;
register_NEML2_object_alias(GaussianScalar, "GaussianScalar");

using GaussianTensor = GaussianTensorTmpl<Tensor>;
register_NEML2_object_alias(GaussianTensor, "GaussianTensor");
} // namespace neml2
