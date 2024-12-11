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

#include "neml2/tensors/LabeledAxis.h"

namespace py = pybind11;
using namespace neml2;

void
def_LabeledAxis(py::module_ & m)
{
  // Accessors, etc.
  // Note that we do not want to expose constructors and modifiers as a neml2::Model typically owns
  // its axes, and so the constructor and modifier bindings introduce ambiguity in ownership, i.e.
  // should C++ or Python own this LabeledAxis?
  auto c = py::class_<LabeledAxis>(m, "LabeledAxis")
               .def("has_item", &LabeledAxis::has_item)
               .def("has_variable",
                    [](const LabeledAxis & self, const LabeledAxisAccessor & name)
                    { return self.has_variable(name); })
               .def("has_subaxis", &LabeledAxis::has_subaxis)
               .def(
                   "subaxis",
                   [](const LabeledAxis & self, const LabeledAxisAccessor & name)
                   { return &self.subaxis(name); },
                   py::return_value_policy::reference)
               .def(
                   "variable_names",
                   [](const LabeledAxis & self, bool recursive)
                   {
                     auto vars = self.sort_by_assembly_order(self.variable_names(recursive));
                     std::vector<std::string> var_names;
                     for (const auto & var : vars)
                       var_names.push_back(utils::stringify(var));
                     return var_names;
                   },
                   py::arg("recursive") = true)
               .def(
                   "subaxis_names",
                   [](const LabeledAxis & self, bool recursive)
                   {
                     auto subaxes_unsrt = self.subaxis_names(recursive);
                     auto subaxes = recursive ? std::vector<LabeledAxisAccessor>(
                                                    subaxes_unsrt.begin(), subaxes_unsrt.end())
                                              : self.sort_by_assembly_order(subaxes_unsrt);
                     std::vector<std::string> subaxis_names;
                     for (const auto & subaxis : subaxes)
                       subaxis_names.push_back(utils::stringify(subaxis));
                     return subaxis_names;
                   },
                   py::arg("recursive") = false)
               .def("storage_size",
                    py::overload_cast<const LabeledAxisAccessor &>(&LabeledAxis::storage_size,
                                                                   py::const_),
                    py::arg("item") = LabeledAxisAccessor())
               .def("nvariable", &LabeledAxis::nvariable, py::arg("recursive") = true)
               .def("nsubaxis", &LabeledAxis::nsubaxis, py::arg("recursive") = false);

  // Operators
  c.def("__repr__", [](const LabeledAxis & self) { return utils::stringify(self); })
      .def("__eq__", [](const LabeledAxis & a, const LabeledAxis & b) { return a == b; })
      .def("__ne__", [](const LabeledAxis & a, const LabeledAxis & b) { return a == b; });
}
