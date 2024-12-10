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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "python/neml2/indexing.h"
#include "python/neml2/types.h"
#include "neml2/tensors/tensors.h"
#include "neml2/misc/math.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(math, m)
{
  m.doc() = "Mathematical functions and utilities";

  // Bring in tensor types
  py::module_::import("neml2.tensors");

  // Methods
  m.def("bmm", &math::bmm);
  m.def("bmv", &math::bmv);
  m.def("bvv", &math::bvv);
  m.def("base_cat", &math::base_cat, py::arg("values"), py::arg("dim") = 0);
  m.def("base_stack", &math::base_stack, py::arg("values"), py::arg("dim") = 0);
  m.def("base_sum", &math::base_sum, py::arg("values"), py::arg("dim") = 0);
  m.def("base_mean", &math::base_mean, py::arg("values"), py::arg("dim") = 0);

  // Templated methods
  // These methods are special because the argument could be anything derived from TensorBase,
  // so we need to bind every possible instantiation.
#define MATH_DEF_TENSORBASE(T)                                                                     \
  m.def("batch_cat", &math::batch_cat<T>, py::arg("values"), py::arg("dim") = 0)                   \
      .def("batch_stack", &math::batch_stack<T>, py::arg("values"), py::arg("dim") = 0)            \
      .def("batch_sum", &math::batch_sum<T>, py::arg("values"), py::arg("dim") = 0)                \
      .def("batch_mean", &math::batch_mean<T>, py::arg("values"), py::arg("dim") = 0)              \
      .def("sign", &math::sign<T>)                                                                 \
      .def("cosh", &math::cosh<T>)                                                                 \
      .def("sinh", &math::sinh<T>)                                                                 \
      .def("tanh", &math::tanh<T>)                                                                 \
      .def("where", &math::where<T>)                                                               \
      .def("heaviside", &math::heaviside<T>)                                                       \
      .def("macaulay", &math::macaulay<T>)                                                         \
      .def("dmacaulay", &math::dmacaulay<T>)                                                       \
      .def("sqrt", &math::sqrt<T>)                                                                 \
      .def("exp", &math::exp<T>)                                                                   \
      .def("abs", &math::abs<T>)                                                                   \
      .def("log", &math::log<T>)

  FOR_ALL_TENSORBASE(MATH_DEF_TENSORBASE);
}
