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

#include "neml2/equation_systems/HVector.h"
#include "neml2/models/Model.h"
#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/core/utils.h"
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<ModelNonlinearSystem> & c)
{
  // wrappers for nonlinear system-related methods
  c.def(py::init<Model *, bool>(),
        py::arg("model"),
        py::arg("assembly_guard") = true,
        "Interpret a Model as a nonlinear system of equations")
      .def("umap", &ModelNonlinearSystem::umap, "Get the unknowns for this nonlinear system")
      .def("unmap", &ModelNonlinearSystem::unmap, "Get the old solutions for this nonlinear system")
      .def("gmap", &ModelNonlinearSystem::gmap, "Get the given variables for this nonlinear system")
      .def("gnmap",
           &ModelNonlinearSystem::gnmap,
           "Get the old given variables for this nonlinear system")
      .def("bmap", &ModelNonlinearSystem::bmap, "Get the residuals for this nonlinear system")
      .def("ulayout",
           &ModelNonlinearSystem::ulayout,
           "Get the unknown layout for this nonlinear system")
      .def("unlayout",
           &ModelNonlinearSystem::unlayout,
           "Get the old solution layout for this nonlinear system")
      .def("glayout",
           &ModelNonlinearSystem::glayout,
           "Get the given variable layout for this nonlinear system")
      .def("gnlayout",
           &ModelNonlinearSystem::gnlayout,
           "Get the old given variable layout for this nonlinear system")
      .def("blayout",
           &ModelNonlinearSystem::blayout,
           "Get the residual layout for this nonlinear system")
      .def("create_uvec",
           &ModelNonlinearSystem::create_uvec,
           "Create a vector for unknowns compatible with this nonlinear system")
      .def("create_unvec",
           &ModelNonlinearSystem::create_unvec,
           "Create a old solution vector compatible with this nonlinear system")
      .def("create_gvec",
           &ModelNonlinearSystem::create_gvec,
           "Create a vector for given variables compatible with this nonlinear system")
      .def("create_gnvec",
           &ModelNonlinearSystem::create_gnvec,
           "Create a vector for old given variables compatible with this nonlinear system")
      .def("create_bvec",
           &ModelNonlinearSystem::create_bvec,
           "Create a vector for residuals compatible with this nonlinear system")
      .def("u_to_un",
           py::overload_cast<const HVector &>(&ModelNonlinearSystem::u_to_un, py::const_),
           py::arg("u"),
           "Map a current unknown vector to the old solution vector")
      .def("g_to_gn",
           py::overload_cast<const HVector &>(&ModelNonlinearSystem::g_to_gn, py::const_),
           py::arg("g"),
           "Map a current given variable vector to the old given variable vector")
      .def("un_to_u",
           py::overload_cast<const HMatrix &>(&ModelNonlinearSystem::un_to_u, py::const_),
           py::arg("A"),
           "Map a matrix whose columns are old solutions to a matrix whose columns are current "
           "unknowns")
      .def("set_u",
           &ModelNonlinearSystem::set_u,
           py::arg("u"),
           "Set the current unknowns in the nonlinear system")
      .def("set_un",
           &ModelNonlinearSystem::set_un,
           py::arg("un"),
           "Set the old solution in the nonlinear system")
      .def("set_g",
           &ModelNonlinearSystem::set_g,
           py::arg("g"),
           "Set the current given variables in the nonlinear system")
      .def("set_gn",
           &ModelNonlinearSystem::set_gn,
           py::arg("gn"),
           "Set the old given variables in the nonlinear system")
      .def("u", &ModelNonlinearSystem::u, "Get the current unknowns from the nonlinear system")
      .def("un", &ModelNonlinearSystem::un, "Get the old solution from the nonlinear system")
      .def("g",
           &ModelNonlinearSystem::g,
           "Get the current given variables from the nonlinear system")
      .def("gn", &ModelNonlinearSystem::gn, "Get the old given variables from the nonlinear system")
      .def("A", &ModelNonlinearSystem::A, "Get the system matrix for the nonlinear system")
      .def("b", &ModelNonlinearSystem::b, "Get the right-hand side vector for the nonlinear system")
      .def("A_and_b",
           &ModelNonlinearSystem::A_and_b,
           "Get the system matrix and the right-hand side vector for the nonlinear system");

  c.def(
      "An",
      [](ModelNonlinearSystem & self)
      {
        if (self.assembly_guard())
          self.model().forward_maybe_jit(false, true, false);
        return self.model().collect_output_derivatives(self.bmap(), self.unmap());
      },
      "Get the derivative of the current residual w.r.t. old solution.");
}
