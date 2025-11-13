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

#include "python/neml2/core/types.h"
#include "python/neml2/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<neml2::TensorValueBase> & c)
{
  c.def(
       "torch",
       [](const TensorValueBase & self) { return at::Tensor(Tensor(self)); },
       "Convert to a torch.Tensor")
      .def(
          "tensor",
          [](const TensorValueBase & self) { return Tensor(self); },
          "Convert to a tensors.Tensor")
      .def_property_readonly(
          "requires_grad",
          [](const TensorValueBase & self) { return Tensor(self).requires_grad(); },
          "Value of the boolean requires_grad property of the underlying tensor.")
      .def(
          "requires_grad_",
          [](TensorValueBase & self, bool req) { return self.requires_grad_(req); },
          py::arg("req") = true,
          "Set the requires_grad property of the underlying tensor.")
      .def(
          "set_",
          [](TensorValueBase & self, const Tensor & val) { self = val; },
          "Modify the underlying tensor data.")
      .def_property_readonly(
          "grad",
          [](const TensorValueBase & self) { return Tensor(self).grad(); },
          "Retrieve the accumulated vector-Jacobian product after a backward propagation.");
}
