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

#include "neml2/tensors/functions/linspace.h"

#include "csrc/math/types.h"
#include <pybind11/cast.h>

namespace py = pybind11;
using namespace neml2;

void
def_linspace(py::module_ & m)
{
#define DEF_LINSPACE(T)                                                                            \
  auto c_##T = py::module_::import("neml2.tensors").attr(#T).cast<py::class_<T>>();                \
  c_##T                                                                                            \
      .def_static("dynamic_linspace",                                                              \
                  py::overload_cast<const T &, const T &, Size, Size>(&dynamic_linspace),          \
                  py::arg("start"),                                                                \
                  py::arg("end"),                                                                  \
                  py::arg("step"),                                                                 \
                  py::arg("dim") = 0)                                                              \
      .def_static("intmd_linspace",                                                                \
                  py::overload_cast<const T &, const T &, Size, Size>(&intmd_linspace),            \
                  py::arg("start"),                                                                \
                  py::arg("end"),                                                                  \
                  py::arg("step"),                                                                 \
                  py::arg("dim") = 0);
  FOR_ALL_TENSORBASE(DEF_LINSPACE);

  c_Tensor.def_static("base_linspace",
                      py::overload_cast<const Tensor &, const Tensor &, Size, Size>(&base_linspace),
                      py::arg("start"),
                      py::arg("end"),
                      py::arg("step"),
                      py::arg("dim") = 0);
}
