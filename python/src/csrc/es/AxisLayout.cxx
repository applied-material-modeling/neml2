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

#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/base/Parser.h"

#include "csrc/es/types.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<AxisLayout> & c)
{
  py::enum_<AxisLayout::IStructure>(c, "IStructure")
      .value("DENSE",
             AxisLayout::IStructure::DENSE,
             "All intermediate dimensions are grouped into base dimensions")
      .value("BLOCK",
             AxisLayout::IStructure::BLOCK,
             "Intermediate dimensions represent blocks of variables")
      .export_values();

  c.def(py::init<>())
      .def(py::init(
               [](const std::vector<std::vector<std::string>> & var_strings,
                  const std::vector<TensorShape> & intmd_shapes,
                  const std::vector<TensorShape> & base_shapes,
                  const std::vector<AxisLayout::IStructure> & istrs)
               {
                 std::vector<std::vector<LabeledAxisAccessor>> vars;
                 vars.reserve(var_strings.size());
                 for (const auto & group : var_strings)
                 {
                   std::vector<LabeledAxisAccessor> cpp_group;
                   cpp_group.reserve(group.size());
                   for (const auto & s : group)
                     cpp_group.emplace_back(utils::parse<VariableName>(s));
                   vars.push_back(std::move(cpp_group));
                 }
                 return AxisLayout(vars, intmd_shapes, base_shapes, istrs);
               }),
           py::arg("vars"),
           py::arg("intmd_shapes"),
           py::arg("base_shapes"),
           py::arg("istrs"),
           R"(
Construct an AxisLayout.

:param vars:         List of variable groups; each group is a list of slash-separated variable names
:param intmd_shapes: Intermediate shape for each variable (flattened across all groups)
:param base_shapes:  Base shape for each variable (flattened across all groups)
:param istrs:        IStructure for each variable group
)")
      .def("ngroup", &AxisLayout::ngroup, "Number of variable groups")
      .def("group_offsets",
           &AxisLayout::group_offsets,
           py::arg("i"),
           "Starting and ending offsets (as a tuple) of variable group i")
      .def("group",
           &AxisLayout::group,
           py::arg("i"),
           py::keep_alive<0, 1>(),
           "Contiguous view of variable group i")
      .def("istr",
           &AxisLayout::istr,
           py::arg("i") = (std::size_t)0,
           "IStructure of variable group i")
      .def(
          "view", &AxisLayout::view, py::keep_alive<0, 1>(), "Contiguous view of the entire layout")
      .def_property_readonly(
          "is_view", &AxisLayout::is_view, "Whether this is a view into a parent layout")
      .def("nvar", &AxisLayout::nvar, "Number of variables")
      .def("storage_sizes",
           &AxisLayout::storage_sizes,
           py::arg("include_intmd"),
           "Storage size of each variable; pass include_intmd=True to include intermediate "
           "dimensions")
      .def(
          "var",
          [](const AxisLayout & self, std::size_t i) { return self.var(i).str(); },
          py::arg("i"),
          "Slash-separated name of variable i")
      .def(
          "intmd_sizes", &AxisLayout::intmd_sizes, py::arg("i"), "Intermediate shape of variable i")
      .def("base_sizes", &AxisLayout::base_sizes, py::arg("i"), "Base shape of variable i")
      .def("__eq__", [](const AxisLayout & a, const AxisLayout & b) { return a == b; });
}
