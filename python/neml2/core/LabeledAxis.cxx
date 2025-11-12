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

#include "neml2/base/LabeledAxis.h"
#include "neml2/base/Parser.h"

#include <pybind11/pybind11.h>

#include "python/neml2/core/types.h"

namespace py = pybind11;
using namespace neml2;

void
def_LabeledAxis(py::module_ & m)
{
  auto c = py::class_<LabeledAxis>(m, "LabeledAxis");

  c.def("has_state", &LabeledAxis::has_state)
      .def("has_old_state", &LabeledAxis::has_old_state)
      .def("has_forces", &LabeledAxis::has_forces)
      .def("has_old_forces", &LabeledAxis::has_old_forces)
      .def("has_residual", &LabeledAxis::has_residual)
      .def("has_parameters", &LabeledAxis::has_parameters)
      .def("size", py::overload_cast<>(&LabeledAxis::size, py::const_))
      .def("size",
           py::overload_cast<const LabeledAxisAccessor &>(&LabeledAxis::size, py::const_),
           py::arg("name"))
      .def("slice", &LabeledAxis::slice, py::arg("name"))
      .def("nvariable", &LabeledAxis::nvariable)
      .def("has_variable", &LabeledAxis::has_variable, py::arg("name"))
      .def("variable_id", &LabeledAxis::variable_id, py::arg("name"))
      .def("variable_names", &LabeledAxis::variable_names)
      .def("variable_slices",
           [](const LabeledAxis & self)
           {
             const auto & slices = self.variable_slices();
             std::vector<py::slice> py_slices;
             py_slices.reserve(slices.size());
             for (const auto & slice : slices)
               py_slices.emplace_back(slice.first, slice.second, 1);
             return py_slices;
           })
      .def(
          "variable_slice",
          [](const LabeledAxis & self, const VariableName & name)
          {
            const auto & slice = self.variable_slice(name);
            return py::slice(slice.first, slice.second, 1);
          },
          py::arg("name"))
      .def("variable_sizes", &LabeledAxis::variable_sizes)
      .def("variable_size", &LabeledAxis::variable_size, py::arg("name"))
      .def("nsubaxis", &LabeledAxis::nsubaxis)
      .def("has_subaxis", &LabeledAxis::has_subaxis, py::arg("name"))
      .def("subaxis_id", &LabeledAxis::subaxis_id, py::arg("name"))
      .def("subaxes", &LabeledAxis::subaxes, py::return_value_policy::reference)
      .def("subaxis",
           py::overload_cast<const LabeledAxisAccessor &>(&LabeledAxis::subaxis, py::const_),
           py::arg("name"),
           py::return_value_policy::reference)
      .def("subaxis_names", &LabeledAxis::subaxis_names)
      .def("subaxis_slices",
           [](const LabeledAxis & self)
           {
             const auto & slices = self.subaxis_slices();
             std::vector<py::slice> py_slices;
             py_slices.reserve(slices.size());
             for (const auto & slice : slices)
               py_slices.emplace_back(slice.first, slice.second, 1);
             return py_slices;
           })
      .def(
          "subaxis_slice",
          [](const LabeledAxis & self, const SubaxisName & name)
          {
            const auto & slice = self.subaxis_slice(name);
            return py::slice(slice.first, slice.second, 1);
          },
          py::arg("name"))
      .def("subaxis_sizes", &LabeledAxis::subaxis_sizes)
      .def("subaxis_size", &LabeledAxis::subaxis_size, py::arg("name"))
      .def("__repr__", [](const LabeledAxis & self) { return utils::stringify(self); })
      .def("__eq__", [](const LabeledAxis & a, const LabeledAxis & b) { return a == b; })
      .def("__ne__", [](const LabeledAxis & a, const LabeledAxis & b) { return a != b; });
}
