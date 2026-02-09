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

#include "neml2/tensors/tensors.h"

#include <pybind11/pytypes.h>

#include "csrc/tensors/PrimitiveTensor.h"

namespace py = pybind11;
using namespace neml2;

template <class T>
void
def_PrimitiveTensor(py::module_ & m, const std::string & type)
{
  auto c = m.attr(type.c_str()).cast<py::class_<T>>();
  c.def(py::init<const ATensor &, Size>(), py::arg("tensor"), py::arg("intmd_dim") = 0);

  // Static methods
  c.def_static(
       "empty",
       [](NEML2_TENSOR_OPTIONS_VARGS) { return T::empty(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::empty(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::zeros(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::zeros(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::ones(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::ones(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes") = TensorShapeRef{},
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](double init, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::full(init, NEML2_TENSOR_OPTIONS); },
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             double init,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return T::full(dynamic_sizes, intmd_sizes, init, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "rand",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return T::rand(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "rand",
          [](TensorShapeRef dynamic_sizes, TensorShapeRef intmd_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return T::rand(dynamic_sizes, intmd_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}

// Explicit template instantiations
#define INSTANTIATE_PRIMITIVETENSOR(T)                                                             \
  template void def_PrimitiveTensor<T>(py::module_ &, const std::string &)
FOR_ALL_PRIMITIVETENSOR(INSTANTIATE_PRIMITIVETENSOR);
