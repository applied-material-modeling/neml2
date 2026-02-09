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

#include "csrc/core/types.h"
#include "csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<neml2::LabeledAxisAccessor> & c)
{
  c.def(py::init<>())
      .def(py::init([](const py::object & str) { return unpack_variable_name(str); }))
      .def(py::init<const LabeledAxisAccessor &>())
      .def("with_suffix", &LabeledAxisAccessor::with_suffix)
      .def("append",
           [](const LabeledAxisAccessor & self, const py::object & other)
           { return self.append(unpack_variable_name(other)); })
      .def("prepend",
           [](const LabeledAxisAccessor & self, const py::object & other)
           { return self.prepend(unpack_variable_name(other)); })
      .def(
          "remount",
          [](const LabeledAxisAccessor & self, const py::object & other, Size n)
          { return self.remount(unpack_variable_name(other), n); },
          py::arg("other"),
          py::arg("n") = 1)
      .def("start_with",
           [](const LabeledAxisAccessor & self, const py::object & other)
           { return self.start_with(unpack_variable_name(other)); })
      .def("current", &LabeledAxisAccessor::current)
      .def("old", &LabeledAxisAccessor::old)
      .def("__repr__", [](const LabeledAxisAccessor & self) { return self.str(); })
      .def("__bool__", [](const LabeledAxisAccessor & self) { return !self.empty(); })
      .def("__len__", [](const LabeledAxisAccessor & self) { return self.size(); })
      .def("__hash__",
           [](const LabeledAxisAccessor & self) { return py::hash(py::cast(self.str())); })
      .def("__eq__",
           [](const LabeledAxisAccessor & a, const py::object & b)
           { return a == unpack_variable_name(b); })
      .def("__ne__",
           [](const LabeledAxisAccessor & a, const py::object & b)
           { return a != unpack_variable_name(b); });
}
