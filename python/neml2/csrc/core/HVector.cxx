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
#include "python/neml2/csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<neml2::HVector> & c)
{
  c.def(py::init<std::vector<TensorShape>>(), py::arg("block_sizes"))
      .def(py::init(
               [](const py::sequence & blocks, const std::vector<TensorShape> & block_sizes)
               {
                 auto base_shape_lookup = [&block_sizes](std::size_t i) -> TensorShapeRef
                 { return block_sizes[i]; };
                 return HVector(unpack_tensor_list(blocks, base_shape_lookup), block_sizes);
               }),
           py::arg("blocks"),
           py::arg("block_sizes"))
      .def_property_readonly("requires_grad", &neml2::HVector::requires_grad)
      .def_property_readonly("dtype",
                             [](const neml2::HVector & self)
                             { return c10::typeMetaToScalarType(self.options().dtype()); })
      .def_property_readonly("device",
                             [](const neml2::HVector & self) { return self.options().device(); })
      .def_property_readonly("n", &neml2::HVector::n)
      .def("zero", &neml2::HVector::zero)
      .def("block_sizes", py::overload_cast<>(&neml2::HVector::block_sizes, py::const_))
      .def("block_sizes",
           py::overload_cast<std::size_t>(&neml2::HVector::block_sizes, py::const_),
           py::arg("i"))
      .def("__getitem__",
           py::overload_cast<std::size_t>(&neml2::HVector::operator[], py::const_),
           py::arg("i"))
      .def("__setitem__", py::overload_cast<std::size_t>(&neml2::HVector::operator[]), py::arg("i"))
      .def("assemble", [](const HVector & self) { return self.assemble(); })
      .def(
          "assemble",
          [](const HVector & self, const std::vector<std::size_t> & blocks)
          { return self.assemble(blocks); },
          py::arg("blocks"))
      .def(
          "disassemble",
          [](HVector & self, const Tensor & assembled) { self.disassemble(assembled); },
          py::arg("assembled"))
      .def(
          "disassemble",
          [](HVector & self, const Tensor & assembled, const std::vector<std::size_t> & blocks)
          { self.disassemble(assembled, blocks); },
          py::arg("assembled"),
          py::arg("blocks"));

  // operators
  c.def(-py::self)
      .def(py::self + py::self)
      .def(Scalar() + py::self)
      .def(py::self + Scalar())
      .def(double() + py::self)
      .def(py::self + double())
      .def(py::self - py::self)
      .def(Scalar() - py::self)
      .def(py::self - Scalar())
      .def(double() - py::self)
      .def(py::self - double())
      .def(py::self * py::self)
      .def(Scalar() * py::self)
      .def(py::self * Scalar())
      .def(double() * py::self)
      .def(py::self * double())
      .def(py::self / Scalar())
      .def(py::self / double());
}
