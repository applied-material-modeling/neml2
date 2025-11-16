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

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<Tensor> & c)
{
  def_TensorBase<Tensor>(m, "Tensor");

  // Static methods
  c.def_static(
       "empty",
       [](TensorShapeRef base_sizes, NEML2_TENSOR_OPTIONS_VARGS)
       { return Tensor::empty(base_sizes, NEML2_TENSOR_OPTIONS); },
       py::arg("base_sizes"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             TensorShapeRef base_sizes,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::empty(dynamic_sizes, intmd_sizes, base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef base_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::zeros(base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             TensorShapeRef base_sizes,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::zeros(dynamic_sizes, intmd_sizes, base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef base_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::ones(base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             TensorShapeRef base_sizes,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::ones(dynamic_sizes, intmd_sizes, base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef base_sizes, double init, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::full(base_sizes, init, NEML2_TENSOR_OPTIONS); },
          py::arg("base_sizes"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             TensorShapeRef base_sizes,
             double init,
             NEML2_TENSOR_OPTIONS_VARGS)
          {
            return Tensor::full(dynamic_sizes, intmd_sizes, base_sizes, init, NEML2_TENSOR_OPTIONS);
          },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("base_sizes"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "rand",
          [](TensorShapeRef base_sizes, NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::rand(base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "random",
          [](TensorShapeRef dynamic_sizes,
             TensorShapeRef intmd_sizes,
             TensorShapeRef base_sizes,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return Tensor::rand(dynamic_sizes, intmd_sizes, base_sizes, NEML2_TENSOR_OPTIONS); },
          py::arg("dynamic_sizes"),
          py::arg("intmd_sizes"),
          py::arg("base_sizes"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}
