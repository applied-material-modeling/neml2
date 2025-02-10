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

#include "python/neml2/tensors/TensorBase.h"

#include "neml2/tensors/macros.h"
#include "neml2/tensors/functions/pow.h"

namespace py = pybind11;
using namespace neml2;

void
def_Tensor(py::class_<Tensor> & c)
{
  // All PrimitiveTensors are convertible to Tensor
#define Tensor_FROM_PRIMITIVETENSOR(T) c.def(py::init<T>());
  FOR_ALL_PRIMITIVETENSOR(Tensor_FROM_PRIMITIVETENSOR);

  // Static methods
  c.def_static(
       "empty",
       [](TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
       { return Tensor::empty(base_shape, NEML2_TENSOR_OPTIONS); },
       py::arg("base_shape"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](TensorShapeRef batch_shape, TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::empty(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::zeros(base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef batch_shape, TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::zeros(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::ones(base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef batch_shape, TensorShapeRef base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::ones(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef base_shape, Real init, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::full(base_shape, init, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef batch_shape,
             TensorShapeRef base_shape,
             Real init,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::full(batch_shape, base_shape, init, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity",
          [](Size n, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::identity(n, NEML2_TENSOR_OPTIONS); },
          py::arg("n"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity",
          [](TensorShapeRef batch_shape, Size n, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::identity(batch_shape, n, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("n"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);

  // Operators
  c.def("__pow__", [](const Tensor & a, const Tensor & b) { return neml2::pow(a, b); });
}
