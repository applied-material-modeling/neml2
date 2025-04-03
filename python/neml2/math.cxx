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

#include "neml2/tensors/functions/bmm.h"
#include "neml2/tensors/functions/bmv.h"
#include "neml2/tensors/functions/bvv.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/stack.h"
#include "neml2/tensors/functions/sum.h"
#include "neml2/tensors/functions/mean.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/functions/cosh.h"
#include "neml2/tensors/functions/sinh.h"
#include "neml2/tensors/functions/tanh.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/log.h"
#include "neml2/tensors/functions/clip.h"

#include "neml2/tensors/mandel_notation.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(math, m)
{
  m.doc() = "Mathematical functions and utilities";

  // Bring in tensor types
  py::module_::import("neml2.tensors");

  // Methods
  m.def("bmm", &neml2::bmm);
  m.def("bmv", &neml2::bmv);
  m.def("bvv", &neml2::bvv);
  m.def("base_cat", &neml2::base_cat, py::arg("values"), py::arg("dim") = 0);
  m.def("base_stack", &neml2::base_stack, py::arg("values"), py::arg("dim") = 0);
  m.def("base_sum", &neml2::base_sum, py::arg("values"), py::arg("dim") = 0);
  m.def("base_mean", &neml2::base_mean, py::arg("values"), py::arg("dim") = 0);

  // Helpers
  m.def("full_to_mandel", &neml2::full_to_mandel, py::arg("full"), py::arg("dim") = 0);
  m.def("mandel_to_full", &neml2::mandel_to_full, py::arg("mandel"), py::arg("dim") = 0);

  // Templated methods
  // These methods are special because the argument could be anything derived from TensorBase,
  // so we need to bind every possible instantiation.
#define MATH_DEF_TENSORBASE(T)                                                                     \
  m.def("batch_cat",                                                                               \
        py::overload_cast<const std::vector<T> &, Size>(&neml2::batch_cat),                        \
        py::arg("values"),                                                                         \
        py::arg("dim") = 0)                                                                        \
      .def("batch_stack",                                                                          \
           py::overload_cast<const std::vector<T> &, Size>(&neml2::batch_stack),                   \
           py::arg("values"),                                                                      \
           py::arg("dim") = 0)                                                                     \
      .def("batch_sum",                                                                            \
           py::overload_cast<const T &, Size>(&neml2::batch_sum),                                  \
           py::arg("values"),                                                                      \
           py::arg("dim") = 0)                                                                     \
      .def("batch_mean",                                                                           \
           py::overload_cast<const T &, Size>(&neml2::batch_mean),                                 \
           py::arg("values"),                                                                      \
           py::arg("dim") = 0)                                                                     \
      .def("sign", py::overload_cast<const T &>(&neml2::sign))                                     \
      .def("cosh", py::overload_cast<const T &>(&neml2::cosh))                                     \
      .def("sinh", py::overload_cast<const T &>(&neml2::sinh))                                     \
      .def("tanh", py::overload_cast<const T &>(&neml2::tanh))                                     \
      .def("where", py::overload_cast<const ATensor &, const T &, const T &>(&neml2::where))       \
      .def("heaviside", py::overload_cast<const T &>(&neml2::heaviside))                           \
      .def("macaulay", py::overload_cast<const T &>(&neml2::macaulay))                             \
      .def("sqrt", py::overload_cast<const T &>(&neml2::sqrt))                                     \
      .def("exp", py::overload_cast<const T &>(&neml2::exp))                                       \
      .def("abs", py::overload_cast<const T &>(&neml2::abs))                                       \
      .def("log", py::overload_cast<const T &>(&neml2::log))                                       \
      .def("clip", py::overload_cast<const T &, const T &, const T &>(&neml2::clip))

  FOR_ALL_TENSORBASE(MATH_DEF_TENSORBASE);
}
