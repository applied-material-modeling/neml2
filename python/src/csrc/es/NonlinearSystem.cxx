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
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"

#include <pybind11/cast.h>

#include "csrc/es/types.h"
#include "csrc/core/types.h"
#include "csrc/tensors/types.h"
#include "macros.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<ModelNonlinearSystem, std::shared_ptr<ModelNonlinearSystem>> & c)
{
  // NonlinearSystem
  c.def(
       "to",
       [](ModelNonlinearSystem * self, NEML2_TENSOR_OPTIONS_VARGS)
       { self->to(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def(
          "model", &ModelNonlinearSystem::model_ptr, "Get the model defining this nonlinear system")
      .def("set_u",
           &ModelNonlinearSystem::set_u,
           py::arg("u"),
           "Set the unknowns for this nonlinear system")
      .def("set_g",
           &ModelNonlinearSystem::set_g,
           py::arg("g"),
           "Set the given variables for this nonlinear system")
      .def("u", &ModelNonlinearSystem::u, "Get the unknowns for this nonlinear system")
      .def("g", &ModelNonlinearSystem::g, "Get the given variables for this nonlinear system")
      .def("A", &ModelNonlinearSystem::A, "Get the system matrix for this nonlinear system")
      .def(
          "b", &ModelNonlinearSystem::b, "Get the right-hand side vector for this nonlinear system")
      .def("A_and_b",
           &ModelNonlinearSystem::A_and_b,
           "Get the system matrix and the right-hand side vector for this nonlinear system")
      .def("A_and_B",
           &ModelNonlinearSystem::A_and_B,
           "Get the system matrix and the auxiliary system matrix for this nonlinear system")
      .def("A_and_B_and_b",
           &ModelNonlinearSystem::A_and_B_and_b,
           "Get the system matrix, auxiliary system matrix, and the right-hand side vector for "
           "this nonlinear system")
      .def("ulayout",
           &ModelNonlinearSystem::ulayout,
           "Get the ID-to-unknown-base-shape mapping for this nonlinear system")
      .def("blayout",
           &ModelNonlinearSystem::blayout,
           "Get the ID-to-RHS-shape mapping for this nonlinear system")
      .def("glayout",
           &ModelNonlinearSystem::glayout,
           "Get the ID-to-given-base-shape mapping for this nonlinear system");

  // Parameter/buffer related methods
  c.def(
       "named_parameters",
       [](ModelNonlinearSystem & self)
       {
         std::map<std::string, TensorValueBase *> params;
         for (auto && [pname, pval] : self.named_parameters())
           params[pname] = pval.get();
         return params;
       },
       py::return_value_policy::reference,
       "Get the model parameters. The keys of the returned dictionary are the parameters' "
       "names.")
      .def(
          "named_buffers",
          [](ModelNonlinearSystem & self)
          {
            std::map<std::string, TensorValueBase *> buffers;
            for (auto && [bname, bval] : self.named_buffers())
              buffers[bname] = bval.get();
            return buffers;
          },
          py::return_value_policy::reference,
          "Get the model buffers. The keys of the returned dictionary are the buffers' names.")
      .def("__getattr__",
           py::overload_cast<const std::string &>(&ModelNonlinearSystem::get_parameter, py::const_),
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("__setattr__",
           &ModelNonlinearSystem::set_parameter,
           "Set the value for a model parameter")
      .def("get_parameter",
           py::overload_cast<const std::string &>(&ModelNonlinearSystem::get_parameter, py::const_),
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("set_parameter",
           &ModelNonlinearSystem::set_parameter,
           "Set the value for a model parameter")
      .def("set_parameters",
           &ModelNonlinearSystem::set_parameters,
           "Set the values for multiple model parameters ");
}
