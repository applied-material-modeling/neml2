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

#include <pybind11/operators.h>

#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<neml2::HMatrix> & c)
{
  c.def(py::init<std::vector<TensorShape>, std::vector<TensorShape>>(),
        py::arg("row_block_sizes"),
        py::arg("col_block_sizes"))
      .def(py::init<std::vector<std::vector<Tensor>>,
                    std::vector<TensorShape>,
                    std::vector<TensorShape>>(),
           py::arg("values"),
           py::arg("row_block_sizes"),
           py::arg("col_block_sizes"))
      .def_property_readonly("requires_grad", &neml2::HMatrix::requires_grad)
      .def_property_readonly("dtype",
                             [](const neml2::HMatrix & self)
                             { return c10::typeMetaToScalarType(self.options().dtype()); })
      .def_property_readonly("device",
                             [](const neml2::HMatrix & self) { return self.options().device(); })
      .def_property_readonly("m", &neml2::HMatrix::m)
      .def_property_readonly("n", &neml2::HMatrix::n)
      .def("zero", &neml2::HMatrix::zero)
      .def("block_row_sizes", py::overload_cast<>(&neml2::HMatrix::block_row_sizes, py::const_))
      .def("block_row_sizes",
           py::overload_cast<std::size_t>(&neml2::HMatrix::block_row_sizes, py::const_),
           py::arg("i"))
      .def("block_col_sizes", py::overload_cast<>(&neml2::HMatrix::block_col_sizes, py::const_))
      .def("block_col_sizes",
           py::overload_cast<std::size_t>(&neml2::HMatrix::block_col_sizes, py::const_),
           py::arg("i"))
      .def("assemble", [](const HMatrix & self) { return self.assemble(); })
      .def(
          "assemble",
          [](const HMatrix & self,
             const std::vector<std::size_t> & row_blocks,
             const std::vector<std::size_t> & col_blocks)
          { return self.assemble(row_blocks, col_blocks); },
          py::arg("row_blocks"),
          py::arg("col_blocks"))
      .def(
          "disassemble",
          [](HMatrix & self, const Tensor & assembled) { self.disassemble(assembled); },
          py::arg("assembled"))
      .def(
          "disassemble",
          [](HMatrix & self,
             const Tensor & assembled,
             const std::vector<std::size_t> & row_blocks,
             const std::vector<std::size_t> & col_blocks)
          { self.disassemble(assembled, row_blocks, col_blocks); },
          py::arg("assembled"),
          py::arg("row_blocks"),
          py::arg("col_blocks"));

  // operators
  c.def(-py::self);
}
