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

#include "python/neml2/csrc/tensors/TensorBase.h"
#include "python/neml2/csrc/tensors/PrimitiveTensor.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<Rot> & c)
{
  def_TensorBase<Rot>(m, "Rot");
  def_PrimitiveTensor<Rot>(m, "Rot");

  c.def_static(
       "identity",
       [](NEML2_TENSOR_OPTIONS_VARGS) { return Rot::identity(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static("fill_euler_angles", &Rot::fill_euler_angles)
      .def_static("fill_matrix", &Rot::fill_matrix)
      .def_static("fill_rodrigues", &Rot::fill_rodrigues)
      .def_static("rotation_from_to", &Rot::rotation_from_to)
      .def_static("axis_angle", &Rot::axis_angle)
      .def_static("axis_angle_standard", &Rot::axis_angle_standard);

  c.def("euler_rodrigues", &Rot::euler_rodrigues)
      .def("deuler_rodrigues", &Rot::deuler_rodrigues)
      .def("rotate", &Rot::rotate)
      .def("drotate", &Rot::drotate)
      .def("shadow", &Rot::shadow)
      .def("dist", &Rot::dist)
      .def("dV", &Rot::dV);

  c.def(py::self * py::self);
}
