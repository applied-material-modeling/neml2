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

#include "neml2/tensors/crystallography.h"

#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/tensors/types.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(crystallography, m)
{
  m.doc() = "Crystallography helper routines";

  py::module_::import("neml2.tensors");

  m.def("misorientation",
        &crystallography::misorientation,
        py::arg("r1"),
        py::arg("r2"),
        py::arg("orbifold") = "1",
        R"(
Calculate the misorientation of two batches of rotations
:param r1:          First batch of rotations
:param r2:          Second batch of rotations
:param orbifold:    String giving the orbifold notation for the symmetry group
:return:            Batch of misorientation angles in radians
  )");

  m.def("move_to_fundamental_zone",
        &crystallography::move_to_fundamental_zone,
        py::arg("r"),
        py::arg("orbifold"),
        py::arg_v("ref", Rot::fill(0.0, 0.0, 0.005), "Rot(torch.tensor([0.0, 0.0, 0.005]), 0)"),
        R"(
Move a collection of orientations to a fundemental zone defined by the crystal symmetry
:param r:           Batch of rotations to move to the fundamental zone
:param orbifold:    String giving the orbifold notation for the symmetry group
:param ref:         Reference orientation to define the fundamental zone
:return:            Batch of rotations moved to the fundamental zone
)");

  m.def(
      "symmetry",
      [](const std::string & orbifold, NEML2_TENSOR_OPTIONS_VARGS)
      { return crystallography::symmetry(orbifold, NEML2_TENSOR_OPTIONS); },
      py::arg("orbifold"),
      py::kw_only(),
      PY_ARG_TENSOR_OPTIONS,
      R"(
Return the symmetry operators for a given symmetry group as a batch of rank two tensors

:param orbifold:    String giving the orbifold notation for the symmetry group
:param dtype:       Floating point scalar type used throughout the model.
:param device:      Device on which the model will be evaluated. All parameters, buffers,
    and custom data are synced to the given device.
:param requires_grad: If true, turn on requires_grad in the resulting tensor
)");
}
