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
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "python/neml2/indexing.h"
#include "python/neml2/types.h"

#include "neml2/base/Factory.h"
#include "neml2/base/Parser.h"
#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"
#include "neml2/misc/string_utils.h"

namespace py = pybind11;
using namespace neml2;

ValueMap
unpack_tensor_map(py::dict pyinputs, const Model * model = nullptr)
{
  std::vector<VariableName> input_names;
  std::vector<Tensor> input_values;
  for (auto && [key, val] : pyinputs)
  {
    try
    {
      input_names.push_back(key.cast<VariableName>());
    }
    catch (py::cast_error &)
    {
      throw py::cast_error("neml2.Model.value: Invalid input key type -- dictionary "
                           "keys must be convertible to neml2.VariableName");
    }

    try
    {
      input_values.push_back(val.cast<Tensor>());
    }
    catch (py::cast_error &)
    {
      if (THPVariable_Check(val.ptr()))
      {
        const auto x = THPVariable_Unpack(val.ptr());
        // If a model is provided, we can look up the variable's shape information
        if (model)
        {
          if (model->input_axis().has_variable(input_names.back()))
          {
            const auto & xvar = model->input_variable(input_names.back());
            const auto batch_dim = x.dim() - xvar.list_dim() - xvar.base_dim();
            input_values.push_back(Tensor(x, batch_dim));
          }
          // If the input variable does not exist, it doesn't matter what we do
          else
            input_values.push_back(Tensor(x, 0));
        }
        // Otherwise, the best we can do is to assume the tensor is flat,
        // i.e., the base dimension is 1
        else
          input_values.push_back(Tensor(x, x.dim() - 1));
      }
      else
        throw py::cast_error(
            "neml2.Model.value: Invalid input value type -- dictionary values must "
            "be neml2.Tensor or torch.Tensor");
    }
  }

  ValueMap inputs;
  for (size_t i = 0; i < input_names.size(); ++i)
    inputs[input_names[i]] = input_values[i];

  return inputs;
}

PYBIND11_MODULE(core, m)
{
  m.doc() = "NEML2 Python bindings";

  py::module_::import("neml2.tensors");

  // "Forward" declarations
  auto axis_accessor_cls = py::class_<LabeledAxisAccessor>(m, "LabeledAxisAccessor");
  auto axis_cls = py::class_<LabeledAxis>(m, "LabeledAxis");
  auto tensor_value_cls =
      py::class_<TensorValueBase>(m,
                                  "TensorValue",
                                  "The interface for working with tensor values (parameters, "
                                  "buffers, etc.) managed by models.");
  auto model_cls =
      py::class_<Model, std::shared_ptr<Model>>(m, "Model", "A thin wrapper around neml2::Model");

  // Factory methods
  m.def("load_input", &load_input, py::arg("path"), py::arg("cli_args") = "", R"(
Parse all options from an input file. Note that Previously loaded input options
will be discarded.

:param path:     Path to the input file to be parsed
:parma cli_args: Additional command-line arguments to pass to the parser
)");
  m.def("reload_input", &reload_input, py::arg("path"), py::arg("cli_args") = "", R"(
Similar to core.load_input, except that this function additionally clears the
factory so that previously retrieved models are deleted.

This function is only needed if you load and evaluate models inside a for-loop,
where it is desirable to deallocate models on-the-fly.

:param path:     Path to the input file to be parsed
:param cli_args: Additional command-line arguments to pass to the parser
)");
  m.def(
      "get_model",
      [](const std::string & model) { return get_model(model); },
      py::arg("model"),
      py::return_value_policy::reference,
      R"(
Create a models.Model from given input options. The input file must have
already been parsed and loaded.

:param model:        Name of the model
)");
  m.def("load_model",
        &load_model,
        py::arg("path"),
        py::arg("model"),
        py::return_value_policy::reference,
        R"(
A convenient function to load an input file and get a model.

This function is equivalent to calling core.load_input followed by
core.get_model. Note that this convenient function does not support passing
additional command-line arguments and will force the creation of a new
models.Model even if one has already been created. Use core.load_input and
core.get_model if you need finer control over the model creation behavior.

:param path:      Path to the input file to be parsed
:param model:     Name of the model
)");
  m.def("reload_model",
        &reload_model,
        py::arg("path"),
        py::arg("model"),
        py::return_value_policy::reference,
        R"(
Similar to core.load_model, except that this function additionally clears the
factory so that previously retrieved models are deleted.

This function is only needed if you load and evaluate models inside a for-loop,
where it is desirable to deallocate models on-the-fly.

:param path:      Path to the input file to be parsed
:param model:     Name of the model
)");
  m.def(
      "diagnose",
      [](const Model & m)
      {
        auto diagnoses = diagnose(m);
        std::vector<std::string> issues;
        for (const auto & diagnosis : diagnoses)
          issues.push_back(diagnosis.what());
        return issues;
      },
      py::arg("model"),
      R"(
Diagnose common issues in model setup. Raises a runtime error including all identified issues, if any.

:param model: Model to be diagnosed
)");

  // neml2.core.LabeledAxisAccessor
  axis_accessor_cls.def(py::init<>())
      .def(py::init([](const std::string & str) { return utils::parse<LabeledAxisAccessor>(str); }))
      .def(py::init<const LabeledAxisAccessor &>())
      .def("with_suffix", &LabeledAxisAccessor::with_suffix)
      .def("start_with", &LabeledAxisAccessor::start_with)
      .def("append", &LabeledAxisAccessor::append)
      .def("prepend", &LabeledAxisAccessor::prepend)
      .def("remount", &LabeledAxisAccessor::remount)
      .def("current", &LabeledAxisAccessor::current)
      .def("old", &LabeledAxisAccessor::old)
      .def("__repr__", [](const LabeledAxisAccessor & self) { return utils::stringify(self); })
      .def("__bool__", [](const LabeledAxisAccessor & self) { return !self.empty(); })
      .def("__len__", [](const LabeledAxisAccessor & self) { return self.size(); })
      .def("__hash__",
           [](const LabeledAxisAccessor & self)
           { return py::hash(py::cast(utils::stringify(self))); })
      .def("__eq__",
           [](const LabeledAxisAccessor & a, const LabeledAxisAccessor & b) { return a == b; })
      .def("__ne__",
           [](const LabeledAxisAccessor & a, const LabeledAxisAccessor & b) { return a == b; });

  // Make LabeledAxisAccessor implicitly convertible from py::str
  py::implicitly_convertible<std::string, LabeledAxisAccessor>();

  // neml2.core.LabeledAxis
  axis_cls.def("has_state", &LabeledAxis::has_state)
      .def("has_old_state", &LabeledAxis::has_old_state)
      .def("has_forces", &LabeledAxis::has_forces)
      .def("has_old_forces", &LabeledAxis::has_old_forces)
      .def("has_residual", &LabeledAxis::has_residual)
      .def("has_parameters", &LabeledAxis::has_parameters)
      .def("size", py::overload_cast<>(&LabeledAxis::size, py::const_))
      .def("size",
           py::overload_cast<const LabeledAxisAccessor &>(&LabeledAxis::size, py::const_),
           py::arg("name"))
      .def("slice", &LabeledAxis::slice, py::arg("name"))
      .def("nvariable", &LabeledAxis::nvariable)
      .def("has_variable", &LabeledAxis::has_variable, py::arg("name"))
      .def("variable_id", &LabeledAxis::variable_id, py::arg("name"))
      .def("variable_names", &LabeledAxis::variable_names)
      .def("variable_slices",
           [](const LabeledAxis & self)
           {
             const auto & slices = self.variable_slices();
             std::vector<py::slice> py_slices;
             for (const auto & slice : slices)
               py_slices.emplace_back(slice.first, slice.second, 1);
             return py_slices;
           })
      .def(
          "variable_slice",
          [](const LabeledAxis & self, const VariableName & name)
          {
            const auto & slice = self.variable_slice(name);
            return py::slice(slice.first, slice.second, 1);
          },
          py::arg("name"))
      .def("variable_sizes", &LabeledAxis::variable_sizes)
      .def("variable_size", &LabeledAxis::variable_size, py::arg("name"))
      .def("nsubaxis", &LabeledAxis::nsubaxis)
      .def("has_subaxis", &LabeledAxis::has_subaxis, py::arg("name"))
      .def("subaxis_id", &LabeledAxis::subaxis_id, py::arg("name"))
      .def("subaxes", &LabeledAxis::subaxes, py::return_value_policy::reference)
      .def("subaxis",
           py::overload_cast<const LabeledAxisAccessor &>(&LabeledAxis::subaxis, py::const_),
           py::arg("name"),
           py::return_value_policy::reference)
      .def("subaxis_names", &LabeledAxis::subaxis_names)
      .def("subaxis_slices",
           [](const LabeledAxis & self)
           {
             const auto & slices = self.subaxis_slices();
             std::vector<py::slice> py_slices;
             for (const auto & slice : slices)
               py_slices.emplace_back(slice.first, slice.second, 1);
             return py_slices;
           })
      .def(
          "subaxis_slice",
          [](const LabeledAxis & self, const SubaxisName & name)
          {
            const auto & slice = self.subaxis_slice(name);
            return py::slice(slice.first, slice.second, 1);
          },
          py::arg("name"))
      .def("subaxis_sizes", &LabeledAxis::subaxis_sizes)
      .def("subaxis_size", &LabeledAxis::subaxis_size, py::arg("name"))
      .def("__repr__", [](const LabeledAxis & self) { return utils::stringify(self); })
      .def("__eq__", [](const LabeledAxis & a, const LabeledAxis & b) { return a == b; })
      .def("__ne__", [](const LabeledAxis & a, const LabeledAxis & b) { return a == b; });

  // neml2.core.TensorValue
  tensor_value_cls
      .def(
          "torch",
          [](const TensorValueBase & self) { return torch::Tensor(Tensor(self)); },
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

  // neml2.core.Model
  model_cls.def_property_readonly("name", &Model::name, "Name of the model")
      .def(
          "to",
          [](Model & self, NEML2_TENSOR_OPTIONS_VARGS) { return self.to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_property_readonly("type", &Model::type, "Type of the model")
      .def("__str__", [](const Model & self) { return utils::stringify(self); })
      .def(
          "input_axis",
          [](const Model & self) { return &self.input_axis(); },
          py::return_value_policy::reference,
          "Input axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "output_axis",
          [](const Model & self) { return &self.output_axis(); },
          py::return_value_policy::reference,
          "Input axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "input_type",
          [](const Model & self, const VariableName & name)
          { return self.input_variable(name).type(); },
          py::arg("variable"),
          "Introspect the underlying tensor type of an input variable. @returns tensors.TensorType")
      .def(
          "output_type",
          [](const Model & self, const VariableName & name)
          { return self.output_variable(name).type(); },
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
            for (auto submodel : self.registered_models())
              submodels[submodel->name()] = submodel;
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
      .def("set_parameters", &Model::set_parameters, "Set the values for multiple model parameters")
      .def("value",
           [](Model & self, py::dict pyinputs)
           { return self.value(unpack_tensor_map(pyinputs, &self)); })
      .def("dvalue",
           [](Model & self, py::dict pyinputs)
           { return self.dvalue(unpack_tensor_map(pyinputs, &self)); })
      .def("d2value",
           [](Model & self, py::dict pyinputs)
           { return self.d2value(unpack_tensor_map(pyinputs, &self)); })
      .def("value_and_dvalue",
           [](Model & self, py::dict pyinputs)
           { return self.value_and_dvalue(unpack_tensor_map(pyinputs, &self)); })
      .def("dvalue_and_d2value",
           [](Model & self, py::dict pyinputs)
           { return self.dvalue_and_d2value(unpack_tensor_map(pyinputs, &self)); })
      .def("value_and_dvalue_and_d2value",
           [](Model & self, py::dict pyinputs)
           { return self.value_and_dvalue_and_d2value(unpack_tensor_map(pyinputs, &self)); })
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

  // neml2.core.VectorAssembler
  py::class_<VectorAssembler>(m, "VectorAssembler")
      .def(py::init<const LabeledAxis &>())
      .def("assemble_by_variable",
           [](const VectorAssembler & self, py::dict py_vals_dict)
           { return self.assemble_by_variable(unpack_tensor_map(py_vals_dict)); })
      .def("split_by_variable", &VectorAssembler::split_by_variable)
      .def("split_by_subaxis", &VectorAssembler::split_by_subaxis);

  // neml2.core.MatrixAssembler
  py::class_<MatrixAssembler>(m, "MatrixAssembler")
      .def(py::init<const LabeledAxis &, const LabeledAxis &>())
      .def("assemble_by_variable",
           [](const MatrixAssembler & self, py::dict py_vals_dict)
           {
             DerivMap vals_dict;
             for (auto && [key, val] : py_vals_dict)
             {
               try
               {
                 vals_dict[key.cast<VariableName>()] = unpack_tensor_map(val.cast<py::dict>());
               }
               catch (py::cast_error &)
               {
                 throw py::cast_error(
                     "neml2.MatrixAssembler.assemble_by_variable: Invalid input value type -- "
                     "dictionary keys must be convertible to neml2.VariableName, and dictionary "
                     "values must be convertible to dict[neml2.VariableName, neml2.Tensor]");
               }
             }
             return self.assemble_by_variable(vals_dict);
           })
      .def("split_by_variable", &MatrixAssembler::split_by_variable)
      .def("split_by_subaxis", &MatrixAssembler::split_by_subaxis);
}
