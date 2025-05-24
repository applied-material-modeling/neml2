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

#pragma once

#include <pybind11/operators.h>

#include "neml2/misc/types.h"
#include "neml2/tensors/Tensor.h"

#include "python/neml2/types.h"

namespace py = pybind11;

namespace neml2
{

// Forward declarations
template <class Derived>
void def_PrimitiveTensor(py::class_<Derived> & c);

} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{

template <class Derived>
void
def_PrimitiveTensor(py::class_<Derived> & c)
{
  // Ctors, conversions, accessors etc.
  c.def(py::init<const torch::Tensor &>())
      .def("torch", [](const Derived & self) { return torch::Tensor(self); })
      .def("tensor", [](const Derived & self) { return Tensor(self); });

  py::implicitly_convertible<Derived, Tensor>();

  // Static methods
  c.def_static(
       "empty",
       [](NEML2_TENSOR_OPTIONS_VARGS) { return Derived::empty(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](TensorShapeRef batch_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::empty(batch_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return Derived::zeros(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](TensorShapeRef batch_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::zeros(batch_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return Derived::ones(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](TensorShapeRef batch_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::ones(batch_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](Real init, NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::full(init, NEML2_TENSOR_OPTIONS); },
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](TensorShapeRef batch_shape, Real init, NEML2_TENSOR_OPTIONS_VARGS)
          { return Derived::full(batch_shape, init, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}

} // namespace neml2
