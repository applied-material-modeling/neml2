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

#include "neml2/models/ParameterStore.h"
#include "neml2/models/NonlinearParameter.h"
#include "neml2/tensors/macros.h"
#include "neml2/models/Variable.h"

namespace neml2
{
ParameterStore::ParameterStore(OptionSet options, NEML2Object * object)
  : _object(object),
    _object_options(std::move(options))
{
}

void
ParameterStore::send_parameters_to(const torch::TensorOptions & options)
{
  for (auto && [name, param] : _param_values)
    param.to_(options);
}

void
ParameterStore::set_parameter(const std::string & name, const Tensor & value)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");

  neml_assert(named_parameters().has_key(name), "There is no parameter named ", name);
  named_parameters()[name] = value;
}

void
ParameterStore::set_parameters(const std::map<std::string, Tensor> & param_values)
{
  for (const auto & [name, value] : param_values)
    set_parameter(name, value);
}

TensorValueBase &
ParameterStore::get_parameter(const std::string & name)
{
  auto * base_ptr = _param_values.query_value(name);
  neml_assert(base_ptr, "Parameter named ", name, " does not exist.");
  return *base_ptr;
}

const TensorValueBase &
ParameterStore::get_parameter(const std::string & name) const
{
  const auto * base_ptr = _param_values.query_value(name);
  neml_assert(base_ptr, "Parameter named ", name, " does not exist.");
  return *base_ptr;
}

const VariableBase *
ParameterStore::nl_param(const std::string & name) const
{
  return _nl_params.count(name) ? _nl_params.at(name) : nullptr;
}

Storage<std::string, TensorValueBase> &
ParameterStore::named_parameters()
{
  neml_assert(_object->host() == _object,
              "named_parameters() should only be called on the host model.");
  return _param_values;
}

std::map<std::string, const VariableBase *>
ParameterStore::named_nonlinear_parameters(bool recursive) const
{
  if (!recursive)
    return _nl_params;

  const auto * model = dynamic_cast<const Model *>(this);
  neml_assert(model, "Only models support recursive nonlinear parameter declaration");

  auto all_nl_params = _nl_params;
  for (auto * submodel : model->registered_models())
  {
    auto sub_nl_params = submodel->named_nonlinear_parameters(true);
    all_nl_params.insert(sub_nl_params.begin(), sub_nl_params.end());
  }
  for (auto && [pname, pmodel] : _nl_param_models)
  {
    auto sub_nl_params = pmodel->named_nonlinear_parameters(true);
    all_nl_params.insert(sub_nl_params.begin(), sub_nl_params.end());
  }
  return all_nl_params;
}

std::map<std::string, Model *>
ParameterStore::named_nonlinear_parameter_models(bool recursive) const
{
  if (!recursive)
    return _nl_param_models;

  const auto * model = dynamic_cast<const Model *>(this);
  neml_assert(model, "Only models support recursive nonlinear parameter declaration");

  auto all_nl_param_models = _nl_param_models;
  for (auto * submodel : model->registered_models())
  {
    auto sub_nl_param_models = submodel->named_nonlinear_parameter_models(true);
    all_nl_param_models.insert(sub_nl_param_models.begin(), sub_nl_param_models.end());
  }
  for (auto && [pname, pmodel] : _nl_param_models)
  {
    auto sub_nl_param_models = pmodel->named_nonlinear_parameter_models(true);
    all_nl_param_models.insert(sub_nl_param_models.begin(), sub_nl_param_models.end());
  }
  return all_nl_param_models;
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name,
                                  const std::string & input_option_name,
                                  bool allow_nonlinear)
{
  if (_object_options.contains<T>(input_option_name))
    return declare_parameter(name, _object_options.get<T>(input_option_name));

  if (_object_options.contains<CrossRef<T>>(input_option_name))
    return declare_parameter_crossref<T>(
        name, _object_options.get<CrossRef<T>>(input_option_name), allow_nonlinear);

  throw NEMLException("Internal error in declare_parameter");
}

template <typename T, typename>
std::vector<const T *>
ParameterStore::declare_parameters(const std::string & name,
                                   const std::string & input_option_name,
                                   bool allow_nonlinear)
{

  if (_object_options.contains<std::vector<T>>(input_option_name))
  {
    const auto vals = _object_options.get<std::vector<T>>(input_option_name);
    std::vector<const T *> params(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i)
      params[i] = &declare_parameter(name + "_" + utils::stringify(i), vals[i]);
    return params;
  }

  if (_object_options.contains<std::vector<CrossRef<T>>>(input_option_name))
  {
    const auto vals = _object_options.get<std::vector<CrossRef<T>>>(input_option_name);
    std::vector<const T *> params(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i)
      params[i] = &declare_parameter_crossref<T>(
          name + "_" + utils::stringify(i), vals[i], allow_nonlinear);
    return params;
  }

  throw NEMLException("Internal error in declare_parameters");
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter_crossref(const std::string & name,
                                           const CrossRef<T> & crossref,
                                           bool allow_nonlinear)
{
  try
  {
    return declare_parameter(name, T(crossref));
  }
  catch (const NEMLException & e1)
  {
    try
    {
      // Handle the case of *nonlinear* parameter.
      // Note that nonlinear parameter should only exist inside a Model.
      auto * model = dynamic_cast<Model *>(this);
      neml_assert(model,
                  "Object '",
                  _object->name(),
                  "' of type ",
                  model->type(),
                  "' is trying to declare a parameter named ",
                  name,
                  ". It is not a plain tensor value nor a cross-referenced parameter value. Hence "
                  "I am guessing you are declaring a *nonlinear* parameter. However, nonlinear "
                  "parameter should only be declared by a model, and this object does not appear "
                  "to be a model.");

      neml_assert(allow_nonlinear,
                  "Model '",
                  _object->name(),
                  "' of type ",
                  model->type(),
                  "' is trying to declare a nonlinear parameter named ",
                  name,
                  "'. However, nonlinear coupling has not been implemented for this parameter. If "
                  "this is intended, please consider opening an issue on GitHub including this "
                  "error message.");

      OptionSet extra_opts;
      extra_opts.set<NEML2Object *>("_host") = model->host();
      const auto & pname = crossref.raw();
      auto & nl_param = Factory::get_object<NonlinearParameter<T>>(
          "Models", pname, extra_opts, /*force_create=*/false);
      model->template declare_input_variable<T>(VariableName(pname).prepend(PARAMETERS));
      _nl_params[name] = &nl_param.param();
      _nl_param_models[name] = &nl_param;
      return nl_param.param().value();
    }
    catch (const NEMLException & e2)
    {
      throw NEMLException(
          "Object '" + _object->name() + "' of type " + _object->type() +
          " is trying to register a parameter named '" + name +
          "'.\n\nParsing it as a plain tensor type failed with message:\n" + e1.what() +
          "\n\nParsing it as a nonlinear parameter failed with message:\n" + e2.what() +
          "\n\nIn addition to the above error messages, make sure you provided the correct "
          "parameter name, option name, and parameter type.");
    }
  }
}

#define PARAMETERSTORE_INTANTIATE_PRIMITIVETENSOR(T)                                               \
  template const T & ParameterStore::declare_parameter<T>(                                         \
      const std::string &, const std::string &, bool);                                             \
  template std::vector<const T *> ParameterStore::declare_parameters<T>(                           \
      const std::string &, const std::string &, bool);                                             \
  template const T & ParameterStore::declare_parameter_crossref<T>(                                \
      const std::string &, const CrossRef<T> &, bool)
FOR_ALL_PRIMITIVETENSOR(PARAMETERSTORE_INTANTIATE_PRIMITIVETENSOR);
} // namespace neml2
