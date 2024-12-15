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

#include "python/neml2/tensors/PrimitiveTensor.h"

namespace py = pybind11;
using namespace neml2;

void
def_SR2(py::class_<SR2> & c)
{
  c.def(py::init<const R2 &>());
  c.def_static(
       "fill",
       [](const Real & a, NEML2_TENSOR_OPTIONS_VARGS)
       { return SR2::fill(a, NEML2_TENSOR_OPTIONS); },
       py::arg("value"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static("fill", py::overload_cast<const Scalar &>(&SR2::fill), py::arg("value"))
      .def_static(
          "fill",
          [](const Real & a11, const Real & a22, const Real & a33, NEML2_TENSOR_OPTIONS_VARGS)
          { return SR2::fill(a11, a22, a33, NEML2_TENSOR_OPTIONS); },
          py::arg("value_00"),
          py::arg("value_11"),
          py::arg("value_22"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("fill",
                  py::overload_cast<const Scalar &, const Scalar &, const Scalar &>(&SR2::fill),
                  py::arg("value_00"),
                  py::arg("value_11"),
                  py::arg("value_22"))
      .def_static(
          "fill",
          [](const Real & a11,
             const Real & a22,
             const Real & a33,
             const Real & a23,
             const Real & a13,
             const Real & a12,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return SR2::fill(a11, a22, a33, a23, a13, a12, NEML2_TENSOR_OPTIONS); },
          py::arg("value_00"),
          py::arg("value_11"),
          py::arg("value_22"),
          py::arg("value_12"),
          py::arg("value_02"),
          py::arg("value_01"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("fill",
                  py::overload_cast<const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &>(&SR2::fill),
                  py::arg("value_00"),
                  py::arg("value_11"),
                  py::arg("value_22"),
                  py::arg("value_12"),
                  py::arg("value_02"),
                  py::arg("value_01"));
}
