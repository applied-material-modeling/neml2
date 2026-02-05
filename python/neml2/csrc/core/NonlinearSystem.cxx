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
#include "neml2/equation_systems/assembly.h"
#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/core/utils.h"
#include <pybind11/cast.h>

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<ModelNonlinearSystem, std::shared_ptr<ModelNonlinearSystem>> & c)
{
  // assembly routines
  m.def(
       "from_assembly",
       [](const Tensor & t, const TensorShape & intmd_shape, const TensorShape & base_shape)
       { return from_assembly<1>(t, {intmd_shape}, {base_shape}); },
       py::arg("from"),
       py::arg("intmd_shape"),
       py::arg("base_shape"),
       "Convert a tensor (variable value) from assembly format to normal format")
      .def(
          "from_assembly",
          [](const Tensor & t,
             const TensorShape & intmd_shape1,
             const TensorShape & intmd_shape2,
             const TensorShape & base_shape1,
             const TensorShape & base_shape2)
          { return from_assembly<2>(t, {intmd_shape1, intmd_shape2}, {base_shape1, base_shape2}); },
          py::arg("from"),
          py::arg("intmd_shape1"),
          py::arg("intmd_shape2"),
          py::arg("base_shape1"),
          py::arg("base_shape2"),
          "Convert a tensor (variable derivative) from assembly format to normal format")
      .def(
          "to_assembly",
          [](const Tensor & t, const TensorShape & intmd_shape, const TensorShape & base_shape)
          { return to_assembly<1>(t, {intmd_shape}, {base_shape}); },
          py::arg("from"),
          py::arg("intmd_shape"),
          py::arg("base_shape"),
          "Convert a tensor (variable value) to assembly format from normal format")
      .def(
          "to_assembly",
          [](const Tensor & t,
             const TensorShape & intmd_shape1,
             const TensorShape & intmd_shape2,
             const TensorShape & base_shape1,
             const TensorShape & base_shape2)
          { return to_assembly<2>(t, {intmd_shape1, intmd_shape2}, {base_shape1, base_shape2}); },
          py::arg("from"),
          py::arg("intmd_shape1"),
          py::arg("intmd_shape2"),
          py::arg("base_shape1"),
          py::arg("base_shape2"),
          "Convert a tensor (variable derivative) to assembly format from normal format")
      .def(
          "assemble_vector",
          [](const std::vector<Tensor> & ts, const std::vector<TensorShape> & base_shapes)
          { return assemble(SparseTensorList(ts), std::nullopt, base_shapes); },
          py::arg("tensors"),
          py::arg("base_shapes"),
          "Assemble a list of tensors into a single tensor (vector, with one base dimension) in "
          "assembly format")
      .def(
          "disassemble_vector",
          [](const Tensor & t, const std::vector<TensorShape> & base_shapes)
          { return std::vector<Tensor>(disassemble(t, std::nullopt, base_shapes)); },
          py::arg("tensor"),
          py::arg("base_shapes"),
          "Disassemble a tensor (vector, with one base dimension) into a list of tensors from "
          "assembly format")
      .def(
          "assemble_matrix",
          [](const std::vector<Tensor> & ts,
             const std::vector<TensorShape> & row_base_shapes,
             const std::vector<TensorShape> & col_base_shapes)
          {
            return assemble(
                SparseTensorList(ts), std::nullopt, std::nullopt, row_base_shapes, col_base_shapes);
          },
          py::arg("tensors"),
          py::arg("row_base_shapes"),
          py::arg("col_base_shapes"),
          "Assemble a list of tensors into a single tensor (matrix, with two base dimensions) in "
          "assembly format")
      .def(
          "disassemble_matrix",
          [](const Tensor & t,
             const std::vector<TensorShape> & row_base_shapes,
             const std::vector<TensorShape> & col_base_shapes)
          {
            return std::vector<Tensor>(
                disassemble(t, std::nullopt, std::nullopt, row_base_shapes, col_base_shapes));
          },
          py::arg("tensor"),
          py::arg("row_base_shapes"),
          py::arg("col_base_shapes"),
          "Disassemble a tensor (matrix, with two base dimensions) into a list of tensors from "
          "assembly format");

  // NonlinearSystem
  c.def(
       "to",
       [](ModelNonlinearSystem * self, NEML2_TENSOR_OPTIONS_VARGS)
       { return self->to(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def(
          "model",
          [](const ModelNonlinearSystem * self) { return self->model_ptr(); },
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
