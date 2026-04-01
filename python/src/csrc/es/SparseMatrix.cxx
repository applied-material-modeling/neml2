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

#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/AssembledMatrix.h"

#include "csrc/es/types.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<SparseMatrix> & c)
{
  c.def(py::init<>())
      .def(py::init<AxisLayout, AxisLayout>(), py::arg("row_layout"), py::arg("col_layout"))
      .def(py::init<AxisLayout, AxisLayout, std::vector<std::vector<Tensor>>>(),
           py::arg("row_layout"),
           py::arg("col_layout"),
           py::arg("tensors"))
      .def_readwrite("tensors",
                     &SparseMatrix::tensors,
                     "2D list of tensors, indexed by [row variable][col variable]")
      .def_readwrite("row_layout", &SparseMatrix::row_layout, "Row layout of the tensors")
      .def_readwrite("col_layout", &SparseMatrix::col_layout, "Column layout of the tensors")
      .def("group",
           &SparseMatrix::group,
           py::arg("i"),
           py::arg("j"),
           "Semi-contiguous view of block (i, j)")
      .def(
          "assemble", &SparseMatrix::assemble, "Assemble the tensors into a dense AssembledMatrix");
}
