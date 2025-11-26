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

#include <pybind11/stl.h>

#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<MatrixAssembler> & c)
{
  c.def(py::init<const LabeledAxis &, const LabeledAxis &>())
      .def(
          "assemble_by_variable",
          [](const MatrixAssembler & self, const py::dict & py_vals_dict, bool assembly)
          {
            auto base_shape_lookup_i =
                [axis = &self.yaxis()](const VariableName & key) -> TensorShapeRef
            {
              const auto vid = axis->variable_id(axis->disqualify(key));
              return axis->variable_base_sizes()[vid];
            };
            auto base_shape_lookup_j =
                [axis = &self.yaxis()](const VariableName & key) -> TensorShapeRef
            {
              const auto vid = axis->variable_id(axis->disqualify(key));
              return axis->variable_base_sizes()[vid];
            };
            const auto vals =
                unpack_deriv_map(py_vals_dict, assembly, base_shape_lookup_i, base_shape_lookup_j);
            return self.assemble_by_variable(vals, assembly);
          },
          py::arg("derivs"),
          py::arg("assembly") = true)
      .def(
          "split_by_variable",
          [](const MatrixAssembler & self, const Tensor & mat, bool assembly)
          { return self.split_by_variable(mat, assembly); },
          py::arg("mat"),
          py::arg("assembly") = true)
      .def("split_by_subaxis", &MatrixAssembler::split_by_subaxis);
}
