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

#include "csrc/core/types.h"
#include "csrc/core/utils.h"

namespace py = pybind11;
using namespace neml2;

void
def(py::module_ & m, py::class_<Model, std::shared_ptr<Model>> & c)
{
  c.def(
       "to",
       [](Model & self, NEML2_TENSOR_OPTIONS_VARGS) { return self.to(NEML2_TENSOR_OPTIONS); },
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_property_readonly("type", &Model::type, "Type of the model")
      .def("__str__", [](const Model & self) { return utils::stringify(self); })
      .def(
          "input_variables",
          [](Model & self)
          {
            std::vector<std::string> vnames;
            for (const auto & [name, var] : self.input_variables())
              vnames.push_back(name.str());
            return vnames;
          },
          "Input variables of the model.")
      .def(
          "output_variables",
          [](Model & self)
          {
            std::vector<std::string> vnames;
            for (const auto & [name, var] : self.output_variables())
              vnames.push_back(name.str());
            return vnames;
          },
          "Output variables of the model.")
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
      .def("zero_undefined_input", &Model::zero_undefined_input, "Zero undefined input variables")
      .def("value",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             return pack_value_map(
                 self.value(unpack_value_map(pyinputs, false, base_shape_lookup)));
           })
      .def("dvalue",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             return pack_deriv_map(
                 self.dvalue(unpack_value_map(pyinputs, false, base_shape_lookup)));
           })
      .def("value_and_dvalue",
           [](Model & self, const py::dict & pyinputs)
           {
             auto base_shape_lookup = [model = &self](const VariableName & key) -> TensorShapeRef
             { return model->input_variable(key).base_sizes(); };
             auto [values, derivs] =
                 self.value_and_dvalue(unpack_value_map(pyinputs, false, base_shape_lookup));
             return std::make_pair(pack_value_map(values), pack_deriv_map(derivs));
           });
}
