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
def(py::module_ & m, py::class_<SSR4> & c)
{
  def_TensorBase<SSR4>(m, "SSR4");
  def_PrimitiveTensor<SSR4>(m, "SSR4");

  c.def_static(
       "identity",
       [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_sym",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_sym(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_vol",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_vol(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_dev",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_dev(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_C1",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_C1(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_C2",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_C2(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity_C3",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return SSR4::identity_C3(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "isotropic_E_nu",
          [](double E, double nu, NEML2_TENSOR_OPTIONS_VARGS)
          { return SSR4::isotropic_E_nu(E, nu, NEML2_TENSOR_OPTIONS); },
          py::arg("E"),
          py::arg("nu"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static("isotropic_E_nu",
                  py::overload_cast<const Scalar &, const Scalar &>(&SSR4::isotropic_E_nu))
      .def_static(
          "fill_C1_C2_C3",
          [](double C1, double C2, double C3, NEML2_TENSOR_OPTIONS_VARGS)
          { return SSR4::fill_C1_C2_C3(C1, C2, C3, NEML2_TENSOR_OPTIONS); },
          py::arg("C1"),
          py::arg("C2"),
          py::arg("C3"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "fill_C1_C2_C3",
          py::overload_cast<const Scalar &, const Scalar &, const Scalar &>(&SSR4::fill_C1_C2_C3));

  // Operators
  c.def(py::self * py::self).def(py::self * SR2());
}
