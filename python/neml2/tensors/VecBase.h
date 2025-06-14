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
#include "neml2/tensors/Rot.h"

#include "python/neml2/types.h"

namespace py = pybind11;

namespace neml2
{

// Forward declarations
template <class Derived>
void def_VecBase(py::class_<Derived> & c);

} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{

template <class Derived>
void
def_VecBase(py::class_<Derived> & c)
{
  // Ctors, conversions, accessors etc.
  c.def("__call__", &Derived::operator());

  // Methods
  c.def("norm_sq", &Derived::norm_sq)
      .def("norm", &Derived::norm)
      .def("rotate", py::overload_cast<const Rot &>(&Derived::rotate, py::const_))
      .def("drotate", py::overload_cast<const Rot &>(&Derived::drotate, py::const_));

  // Static methods
  c.def_static(
       "fill",
       [](const double & v1, const double & v2, const double & v3, NEML2_TENSOR_OPTIONS_VARGS)
       { return Derived::fill(v1, v2, v3, NEML2_TENSOR_OPTIONS); },
       py::arg("x"),
       py::arg("y"),
       py::arg("z"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "fill",
          [](const Scalar & v1, const Scalar & v2, const Scalar & v3)
          { return Derived::fill(v1, v2, v3); },
          py::arg("x"),
          py::arg("y"),
          py::arg("z"))
      .def_static(
          "identity_map",
          [](NEML2_TENSOR_OPTIONS_VARGS) { return Derived::identity_map(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}

} // namespace neml2
