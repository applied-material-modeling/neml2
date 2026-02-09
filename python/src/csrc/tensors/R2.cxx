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

#include <pybind11/operators.h>

#include "csrc/tensors/TensorBase.h"
#include "csrc/tensors/PrimitiveTensor.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<R2> & c)
{
  def_TensorBase<R2>(m, "R2");
  def_PrimitiveTensor<R2>(m, "R2");

  c.def_static(
       "fill",
       [](double a, NEML2_TENSOR_OPTIONS_VARGS) { return SR2::fill(a, NEML2_TENSOR_OPTIONS); },
       py::arg("a"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static("fill", py::overload_cast<const Scalar &>(&SR2::fill))
      .def_static(
          "fill",
          [](double a11, double a22, double a33, NEML2_TENSOR_OPTIONS_VARGS)
          { return SR2::fill(a11, a22, a33, NEML2_TENSOR_OPTIONS); },
          py::arg("a11"),
          py::arg("a22"),
          py::arg("a33"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("fill",
                  py::overload_cast<const Scalar &, const Scalar &, const Scalar &>(&SR2::fill))
      .def_static(
          "fill",
          [](double a11,
             double a22,
             double a33,
             double a23,
             double a13,
             double a12,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return SR2::fill(a11, a22, a33, a23, a13, a12, NEML2_TENSOR_OPTIONS); },
          py::arg("a11"),
          py::arg("a22"),
          py::arg("a33"),
          py::arg("a23"),
          py::arg("a13"),
          py::arg("a12"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("fill",
                  py::overload_cast<const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &,
                                    const Scalar &>(&SR2::fill))
      .def_static(
          "fill",
          [](double a11,
             double a12,
             double a13,
             double a21,
             double a22,
             double a23,
             double a31,
             double a32,
             double a33,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return R2::fill(a11, a12, a13, a21, a22, a23, a31, a32, a33, NEML2_TENSOR_OPTIONS); },
          py::arg("a11"),
          py::arg("a12"),
          py::arg("a13"),
          py::arg("a21"),
          py::arg("a22"),
          py::arg("a23"),
          py::arg("a31"),
          py::arg("a32"),
          py::arg("a33"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("fill",
                  [](const Scalar & a11,
                     const Scalar & a12,
                     const Scalar & a13,
                     const Scalar & a21,
                     const Scalar & a22,
                     const Scalar & a23,
                     const Scalar & a31,
                     const Scalar & a32,
                     const Scalar & a33)
                  { return R2::fill(a11, a12, a13, a21, a22, a23, a31, a32, a33); });

  c.def(py::self * py::self).def(py::self * Vec());
}
