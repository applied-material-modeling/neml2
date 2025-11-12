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
// #include <pybind11/stl.h>
// #include <pybind11/stl/filesystem.h>

#include "python/neml2/core/LabeledAxisAccessor.h"
#include "python/neml2/core/LabeledAxis.h"

using namespace neml2;

// static ValueMap
// unpack_tensor_map(const py::dict & pyinputs, const Model * model = nullptr)
// {
//   std::vector<VariableName> input_names;
//   std::vector<Tensor> input_values;
//   for (auto && [key, val] : pyinputs)
//   {
//     try
//     {
//       input_names.push_back(key.cast<VariableName>());
//     }
//     catch (py::cast_error &)
//     {
//       throw py::cast_error("neml2.Model.value: Invalid input key -- dictionary keys must be "
//                            "convertible to neml2.VariableName");
//     }

//     try
//     {
//       input_values.push_back(val.cast<Tensor>());
//     }
//     catch (py::cast_error &)
//     {
//       throw py::cast_error("neml2.Model.value: Invalid input value for variable '" +
//                            input_names.back().str() +
//                            "' -- dictionary values must "
//                            "be convertible to neml2.Tensor");
//     }
//   }

//   ValueMap inputs;
//   for (size_t i = 0; i < input_names.size(); ++i)
//     inputs[input_names[i]] = input_values[i];

//   return inputs;
// }

PYBIND11_MODULE(core, m)
{
  m.doc() = "NEML2 Python bindings";

  // py::module_::import("neml2.tensors");

  // "Forward" declarations
  // auto axis_accessor_cls = py::class_<LabeledAxisAccessor>(m, "LabeledAxisAccessor");
  // auto axis_cls = py::class_<LabeledAxis>(m, "LabeledAxis");
  // auto tensor_value_cls =
  //     py::class_<TensorValueBase>(m,
  //                                 "TensorValue",
  //                                 "The interface for working with tensor values (parameters, "
  //                                 "buffers, etc.) managed by models.");
  // auto factory_cls =
  //     py::class_<Factory>(m, "Factory", "Factory for creating objects defined in the input
  //     file");
  // auto model_cls =
  //     py::class_<Model, std::shared_ptr<Model>>(m, "Model", "A thin wrapper around
  //     neml2::Model");

  // Factory methods
  //   m.def("load_input",
  //         &load_input,
  //         py::arg("path"),
  //         py::arg("cli_args") = "",
  //         R"(
  // Parse all options from an input file. Note that Previously loaded input options
  // will be discarded.

  // :param path:     Path to the input file to be parsed
  // :param cli_args: Additional command-line arguments to pass to the parser
  // )");
  //   factory_cls.def(
  //       "get_model",
  //       [](Factory * self, const std::string & name) { return self->get_model(name); },
  //       py::arg("name"),
  //       R"(
  // Create a core.Model.

  // :param name:        Name of the model
  // )");
  //   m.def("load_model",
  //         &load_model,
  //         py::arg("path"),
  //         py::arg("name"),
  //         R"(
  // A convenient function to load an input file and get a model.

  // This function is equivalent to calling core.load_input followed by
  // Factory.get_model. Note that this convenient function does not support passing
  // additional command-line arguments and will force the creation of a new
  // core.Model even if one has already been created. Use core.load_input and
  // Factory.get_model if you need finer control over the model creation behavior.

  // :param path:      Path to the input file to be parsed
  // :param name:      Name of the model
  // )");
  //   m.def(
  //       "diagnose",
  //       [](const Model & m)
  //       {
  //         auto diagnoses = diagnose(m);
  //         std::vector<std::string> issues;
  //         issues.reserve(diagnoses.size());
  //         for (const auto & diagnosis : diagnoses)
  //           issues.emplace_back(diagnosis.what());
  //         return issues;
  //       },
  //       py::arg("model"),
  //       R"(
  // Diagnose common issues in model setup. Raises a runtime error including all identified issues,
  // if any.

  // :param model: Model to be diagnosed
  // )");

  // binding definitions
  def_LabeledAxisAccessor(m);
  def_LabeledAxis(m);

  // neml2.core.TensorValue
  // tensor_value_cls
  //     .def(
  //         "torch",
  //         [](const TensorValueBase & self) { return torch::Tensor(Tensor(self)); },
  //         "Convert to a torch.Tensor")
  //     .def(
  //         "tensor",
  //         [](const TensorValueBase & self) { return Tensor(self); },
  //         "Convert to a tensors.Tensor")
  //     .def_property_readonly(
  //         "requires_grad",
  //         [](const TensorValueBase & self) { return Tensor(self).requires_grad(); },
  //         "Value of the boolean requires_grad property of the underlying tensor.")
  //     .def(
  //         "requires_grad_",
  //         [](TensorValueBase & self, bool req) { return self.requires_grad_(req); },
  //         py::arg("req") = true,
  //         "Set the requires_grad property of the underlying tensor.")
  //     .def(
  //         "set_",
  //         [](TensorValueBase & self, const Tensor & val) { self = val; },
  //         "Modify the underlying tensor data.")
  //     .def_property_readonly(
  //         "grad",
  //         [](const TensorValueBase & self) { return Tensor(self).grad(); },
  //         "Retrieve the accumulated vector-Jacobian product after a backward propagation.");

  // neml2.core.Model
  // model_cls.def_property_readonly("name", &Model::name, "Name of the model")
  //     .def(
  //         "to",
  //         [](Model & self, NEML2_TENSOR_OPTIONS_VARGS) { return self.to(NEML2_TENSOR_OPTIONS); },
  //         py::kw_only(),
  //         PY_ARG_TENSOR_OPTIONS)
  //     .def_property_readonly("type", &Model::type, "Type of the model")
  //     .def("__str__", [](const Model & self) { return utils::stringify(self); })
  //     .def(
  //         "input_axis",
  //         [](const Model & self) { return &self.input_axis(); },
  //         py::return_value_policy::reference,
  //         "Input axis of the model. The axis contains information on variable names and their "
  //         "associated slicing indices.")
  //     .def(
  //         "output_axis",
  //         [](const Model & self) { return &self.output_axis(); },
  //         py::return_value_policy::reference,
  //         "Input axis of the model. The axis contains information on variable names and their "
  //         "associated slicing indices.")
  //     .def(
  //         "input_type",
  //         [](const Model & self, const VariableName & name)
  //         { return self.input_variable(name).type(); },
  //         py::arg("variable"),
  //         "Introspect the underlying tensor type of an input variable. @returns
  //         tensors.TensorType")
  //     .def(
  //         "output_type",
  //         [](const Model & self, const VariableName & name)
  //         { return self.output_variable(name).type(); },
  //         py::arg("variable"),
  //         "Introspect the underlying tensor type of an output variable. @returns "
  //         "tensors.TensorType")
  //     .def(
  //         "named_parameters",
  //         [](Model & self)
  //         {
  //           std::map<std::string, TensorValueBase *> params;
  //           for (auto && [pname, pval] : self.named_parameters())
  //             params[pname] = pval.get();
  //           return params;
  //         },
  //         py::return_value_policy::reference,
  //         "Get the model parameters. The keys of the returned dictionary are the parameters' "
  //         "names.")
  //     .def(
  //         "named_buffers",
  //         [](Model & self)
  //         {
  //           std::map<std::string, TensorValueBase *> buffers;
  //           for (auto && [bname, bval] : self.named_buffers())
  //             buffers[bname] = bval.get();
  //           return buffers;
  //         },
  //         py::return_value_policy::reference,
  //         "Get the model buffers. The keys of the returned dictionary are the buffers' names.")
  //     .def(
  //         "named_submodels",
  //         [](const Model & self)
  //         {
  //           std::map<std::string, Model *> submodels;
  //           for (const auto & submodel : self.registered_models())
  //             submodels[submodel->name()] = submodel.get();
  //           return submodels;
  //         },
  //         py::return_value_policy::reference,
  //         "Get the sub-models registered to this model")
  //     .def("__getattr__",
  //          py::overload_cast<const std::string &>(&Model::get_parameter, py::const_),
  //          py::return_value_policy::reference,
  //          "Get a model parameter given its name")
  //     .def("__setattr__", &Model::set_parameter, "Set the value for a model parameter")
  //     .def("get_parameter",
  //          py::overload_cast<const std::string &>(&Model::get_parameter, py::const_),
  //          py::return_value_policy::reference,
  //          "Get a model parameter given its name")
  //     .def("set_parameter", &Model::set_parameter, "Set the value for a model parameter")
  //     .def("set_parameters", &Model::set_parameters, "Set the values for multiple model
  //     parameters") .def("value",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.value(unpack_tensor_map(pyinputs, &self)); })
  //     .def("dvalue",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.dvalue(unpack_tensor_map(pyinputs, &self)); })
  //     .def("d2value",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.d2value(unpack_tensor_map(pyinputs, &self)); })
  //     .def("value_and_dvalue",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.value_and_dvalue(unpack_tensor_map(pyinputs, &self)); })
  //     .def("dvalue_and_d2value",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.dvalue_and_d2value(unpack_tensor_map(pyinputs, &self)); })
  //     .def("value_and_dvalue_and_d2value",
  //          [](Model & self, const py::dict & pyinputs)
  //          { return self.value_and_dvalue_and_d2value(unpack_tensor_map(pyinputs, &self)); })
  //     .def(
  //         "dependency",
  //         [](const Model & self)
  //         {
  //           std::map<std::string, const Model *> deps;
  //           for (auto && [name, var] : self.input_variables())
  //             if (var->ref() != var.get())
  //               deps[utils::stringify(name)] = &var->ref()->owner();
  //           return deps;
  //         },
  //         py::return_value_policy::reference,
  //         "Get the dictionary describing this model's dependency information, if any.");

  // neml2.core.VectorAssembler
  // py::class_<VectorAssembler>(m, "VectorAssembler")
  //     .def(py::init<const LabeledAxis &>())
  //     .def("assemble_by_variable",
  //          [](const VectorAssembler & self, const py::dict & py_vals_dict)
  //          { return self.assemble_by_variable(unpack_tensor_map(py_vals_dict)); })
  //     .def("split_by_variable", &VectorAssembler::split_by_variable)
  //     .def("split_by_subaxis", &VectorAssembler::split_by_subaxis);

  // // neml2.core.MatrixAssembler
  // py::class_<MatrixAssembler>(m, "MatrixAssembler")
  //     .def(py::init<const LabeledAxis &, const LabeledAxis &>())
  //     .def("assemble_by_variable",
  //          [](const MatrixAssembler & self, const py::dict & py_vals_dict)
  //          {
  //            DerivMap vals_dict;
  //            for (auto && [key, val] : py_vals_dict)
  //            {
  //              try
  //              {
  //                vals_dict[key.cast<VariableName>()] = unpack_tensor_map(val.cast<py::dict>());
  //              }
  //              catch (py::cast_error &)
  //              {
  //                throw py::cast_error(
  //                    "neml2.MatrixAssembler.assemble_by_variable: Invalid input value type -- "
  //                    "dictionary keys must be convertible to neml2.VariableName, and dictionary "
  //                    "values must be convertible to dict[neml2.VariableName, neml2.Tensor]");
  //              }
  //            }
  //            return self.assemble_by_variable(vals_dict);
  //          })
  //     .def("split_by_variable", &MatrixAssembler::split_by_variable)
  //     .def("split_by_subaxis", &MatrixAssembler::split_by_subaxis);
}
