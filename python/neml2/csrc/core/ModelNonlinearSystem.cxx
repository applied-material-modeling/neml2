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

#include "neml2/models/ModelNonlinearSystem.h"

#include "neml2/equation_systems/SparseTensorList.h"
#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/core/utils.h"
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<ModelNonlinearSystem> & c)
{
  c.def(
       "to",
       [](ModelNonlinearSystem * self, NEML2_TENSOR_OPTIONS_VARGS)
       { return self->to(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def("model",
           (py::overload_cast<>(&ModelNonlinearSystem::model)),
           py::return_value_policy::reference,
           "Get the model defining this nonlinear system")
      .def_property_readonly(
          "m", &ModelNonlinearSystem::m, "Number of rows of blocks in the system")
      .def_property_readonly(
          "n", &ModelNonlinearSystem::n, "Number of columns of blocks in the system")
      .def_property_readonly(
          "p", &ModelNonlinearSystem::p, "Number of columns of blocks in the auxiliary system")
      .def(
          "set_u",
          [](ModelNonlinearSystem * self, const std::vector<Tensor> & u)
          { self->set_u(SparseTensorList(u)); },
          py::arg("u"),
          "Set the unknowns for this nonlinear system")
      .def(
          "set_g",
          [](ModelNonlinearSystem * self, const std::vector<Tensor> & g)
          { self->set_g(SparseTensorList(g)); },
          py::arg("g"),
          "Set the given variables for this nonlinear system")
      .def(
          "u",
          [](ModelNonlinearSystem * self) { return std::vector<Tensor>(self->u()); },
          "Get the unknowns for this nonlinear system")
      .def(
          "g",
          [](ModelNonlinearSystem * self) { return std::vector<Tensor>(self->g()); },
          "Get the given variables for this nonlinear system")
      .def(
          "A",
          [](ModelNonlinearSystem * self) { return std::vector<Tensor>(self->A()); },
          "Get the system matrix for this nonlinear system")
      .def(
          "b",
          [](ModelNonlinearSystem * self) { return std::vector<Tensor>(self->b()); },
          "Get the right-hand side vector for this nonlinear system")
      .def(
          "A_and_b",
          [](ModelNonlinearSystem * self)
          {
            auto [A, b] = self->A_and_b();
            return std::make_pair(std::vector<Tensor>(A), std::vector<Tensor>(b));
          },
          "Get the system matrix and the right-hand side vector for this nonlinear system")
      .def(
          "A_and_B",
          [](ModelNonlinearSystem * self)
          {
            auto [A, B] = self->A_and_B();
            return std::make_pair(std::vector<Tensor>(A), std::vector<Tensor>(B));
          },
          "Get the auxiliary system matrix for this nonlinear system")
      .def(
          "A_and_B_and_b",
          [](ModelNonlinearSystem * self)
          {
            auto [A, B, b] = self->A_and_B_and_b();
            return std::make_tuple(
                std::vector<Tensor>(A), std::vector<Tensor>(B), std::vector<Tensor>(b));
          },
          "Get the auxiliary system matrix for this nonlinear system")
      .def("umap",
           &ModelNonlinearSystem::umap,
           "Get the ID-to-unknown mapping for this nonlinear system")
      .def("bmap",
           &ModelNonlinearSystem::bmap,
           "Get the ID-to-RHS mapping for this nonlinear system")
      .def("gmap",
           &ModelNonlinearSystem::gmap,
           "Get the ID-to-given mapping for this nonlinear system")
      .def("intmd_ulayout",
           &ModelNonlinearSystem::intmd_ulayout,
           "Get the ID-to-unknown-intermediate-shape mapping for this nonlinear system")
      .def("ulayout",
           &ModelNonlinearSystem::ulayout,
           "Get the ID-to-unknown-base-shape mapping for this nonlinear system")
      .def("intmd_blayout",
           &ModelNonlinearSystem::intmd_blayout,
           "Get the ID-to-RHS-intermediate-shape mapping for this nonlinear system")
      .def("blayout",
           &ModelNonlinearSystem::blayout,
           "Get the ID-to-RHS-shape mapping for this nonlinear system")
      .def("intmd_glayout",
           &ModelNonlinearSystem::intmd_glayout,
           "Get the ID-to-given-intermediate-shape mapping for this nonlinear system")
      .def("glayout",
           &ModelNonlinearSystem::glayout,
           "Get the ID-to-given-base-shape mapping for this nonlinear system");
}
