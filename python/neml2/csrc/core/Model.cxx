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

#include "neml2/base/Parser.h"

#include "python/neml2/csrc/core/types.h"
#include "python/neml2/csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<Model, std::shared_ptr<Model>> & c)
{
  c.def_property_readonly("name", &Model::name, "Name of the model")
      .def(
          "to",
          [](Model & self, NEML2_TENSOR_OPTIONS_VARGS) { return self.to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_property_readonly("type", &Model::type, "Type of the model")
      .def("__str__", [](const Model & self) { return utils::stringify(self); })
      .def(
          "input_axis",
          [](Model & self) { return &self.input_axis(); },
          py::return_value_policy::reference,
          "Input axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "output_axis",
          [](Model & self) { return &self.output_axis(); },
          py::return_value_policy::reference,
          "Output axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "input_type",
          [](const Model & self, const std::string & name)
          { return self.input_variable(utils::parse<VariableName>(name)).type(); },
          py::arg("variable"),
          "Introspect the underlying tensor type of an input variable. @returns tensors.TensorType")
      .def(
          "output_type",
          [](const Model & self, const std::string & name)
          { return self.output_variable(utils::parse<VariableName>(name)).type(); },
          py::arg("variable"),
          "Introspect the underlying tensor type of an output variable. @returns "
          "tensors.TensorType")
      .def(
          "named_parameters",
          [](Model & self)
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
          [](Model & self)
          {
            std::map<std::string, TensorValueBase *> buffers;
            for (auto && [bname, bval] : self.named_buffers())
              buffers[bname] = bval.get();
            return buffers;
          },
          py::return_value_policy::reference,
          "Get the model buffers. The keys of the returned dictionary are the buffers' names.")
      .def(
          "named_submodels",
          [](const Model & self)
          {
            std::map<std::string, Model *> submodels;
            for (const auto & submodel : self.registered_models())
              submodels[submodel->name()] = submodel.get();
            return submodels;
          },
          py::return_value_policy::reference,
          "Get the sub-models registered to this model")
      .def("__getattr__",
           py::overload_cast<const std::string &>(&Model::get_parameter, py::const_),
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("__setattr__", &Model::set_parameter, "Set the value for a model parameter")
      .def("get_parameter",
           py::overload_cast<const std::string &>(&Model::get_parameter, py::const_),
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("set_parameter", &Model::set_parameter, "Set the value for a model parameter")
      .def(
          "set_parameters", &Model::set_parameters, "Set the values for multiple model parameters ")
      .def("value",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             const auto out = self.value(unpack_value_map(pyinputs, false, base_shape_lookup));
             return pack_value_map(out);
           })
      .def("dvalue",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             const auto dout = self.dvalue(unpack_value_map(pyinputs, false, base_shape_lookup));
             return pack_deriv_map(dout);
           })
      .def("value_and_dvalue",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             const auto [out, dout] =
                 self.value_and_dvalue(unpack_value_map(pyinputs, false, base_shape_lookup));
             return std::make_pair(pack_value_map(out), pack_deriv_map(dout));
           })
      .def(
          "assign_input",
          [](Model & self, const py::dict & pyinputs, bool assembly)
          {
            auto base_shape_lookup = [&self](const VariableName & key) -> TensorShapeRef
            { return self.input_variable(key).base_sizes(); };
            return self.assign_input(unpack_value_map(pyinputs, assembly, base_shape_lookup));
          },
          py::arg("inputs"),
          py::arg("assembly") = false)
      .def("residual",
           [](Model & self, const Tensor & x)
           {
             auto r = self.residual(NonlinearSystem::Sol<false>(x));
             return Tensor(r);
           })
      .def("Jacobian",
           [](Model & self, const Tensor & x)
           {
             auto J = self.Jacobian(NonlinearSystem::Sol<false>(x));
             return Tensor(J);
           })
      .def("residual_and_Jacobian",
           [](Model & self, const Tensor & x)
           {
             auto [r, J] = self.residual_and_Jacobian(NonlinearSystem::Sol<false>(x));
             return std::make_pair(Tensor(r), Tensor(J));
           })
      .def(
          "dependency",
          [](const Model & self)
          {
            std::map<std::string, const Model *> deps;
            for (auto && [name, var] : self.input_variables())
              if (var->ref() != var.get())
                deps[utils::stringify(name)] = &var->ref()->owner();
            return deps;
          },
          py::return_value_policy::reference,
          "Get the dictionary describing this model's dependency information, if any.");
}
