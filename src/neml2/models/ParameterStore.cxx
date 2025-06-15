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
#include "neml2/models/Model.h"
#include "neml2/models/InputParameter.h"
#include "neml2/misc/assertions.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/Parser.h"
#include "neml2/base/Settings.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/TensorValue.h"

namespace neml2
{
ParameterStore::ParameterStore(Model * object)
  : _object(object)
{
}

void
ParameterStore::send_parameters_to(const TensorOptions & options)
{
  for (auto && [name, param] : _param_values)
    param->to_(options);
}

void
ParameterStore::set_parameter(const std::string & name, const Tensor & value)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");
  neml_assert(named_parameters().count(name), "There is no parameter named ", name);
  *named_parameters()[name] = value;
}

TensorValueBase &
ParameterStore::get_parameter(const std::string & name)
{
  neml_assert(_object->host() == _object, "This method should only be called on the host model.");
  neml_assert(_param_values.count(name), "Parameter named ", name, " does not exist.");
  return *_param_values[name];
}

void
ParameterStore::set_parameters(const std::map<std::string, Tensor> & param_values)
{
  for (const auto & [name, value] : param_values)
    set_parameter(name, value);
}

std::map<std::string, std::unique_ptr<TensorValueBase>> &
ParameterStore::named_parameters()
{
  neml_assert(_object->host() == _object,
              "named_parameters() should only be called on the host model.");
  return _param_values;
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name, const T & rawval)
{
  if (_object->host() != _object)
    return _object->host<ParameterStore>()->declare_parameter(
        _object->name() + _object->settings().parameter_name_separator() + name, rawval);

  TensorValueBase * base_ptr = nullptr;

  // If the parameter already exists, get it
  if (_param_values.count(name))
    base_ptr = &get_parameter(name);
  // If the parameter doesn't exist, create it
  else
  {
    auto val = std::make_unique<TensorValue<T>>(rawval);
    auto [it, success] = _param_values.emplace(name, std::move(val));
    base_ptr = it->second.get();
  }

  auto ptr = dynamic_cast<TensorValue<T> *>(base_ptr);
  neml_assert(ptr, "Internal error: Failed to cast parameter to a concrete type.");
  return ptr->value();
}

template <typename T>
const T &
resolve_tensor_name(const TensorName<T> & tn, Model * caller, const std::string & pname)
{
  if (!caller)
    throw ParserException("A non-nullptr caller must be provided to resolve a tensor name");

  if constexpr (std::is_same_v<T, ATensor> || std::is_same_v<T, Tensor>)
  {
    throw ParserException("ATensr and Tensor cannot be resolved to a model output variable");
  }
  else
  {
    // When we retrieve a model, we want it to register its own parameters and buffers in the
    // host of the caller.
    OptionSet extra_opts;
    extra_opts.set<NEML2Object *>("_host") = caller->host();

    // The raw string is interpreted as a _variable specifier_ which takes three possible forms
    // 1. "model_name.variable_name"
    // 2. "model_name"
    // 3. "variable_name"
    std::shared_ptr<Model> provider = nullptr;
    VariableName var_name;

    // Split the raw string into tokens with the delimiter '.'
    // There must be either one or two tokens
    auto tokens = utils::split(tn.raw(), ".");
    if (tokens.size() != 1 && tokens.size() != 2)
      throw ParserException("Invalid variable specifier '" + tn.raw() +
                            "'. It should take the form 'model_name', 'variable_name', or "
                            "'model_name.variable_name'");

    // When there is only one token, it must be a model name, and the model must have one and only
    // one output variable.
    if (tokens.size() == 1)
    {
      // Try to parse it as a model name
      const auto & mname = tokens[0];
      try
      {
        // Get the model
        provider = caller->factory()->get_object<Model>(
            "Models", mname, extra_opts, /*force_create=*/false);

        // Apparently, the model must have one and only one output variable.
        const auto nvar = provider->output_axis().nvariable();
        if (nvar == 0)
          throw ParserException(
              "Invalid variable specifier '" + tn.raw() +
              "' (interpreted as model name). The model does not define any output variable.");
        if (nvar > 1)
          throw ParserException(
              "Invalid variable specifier '" + tn.raw() +
              "' (interpreted as model name). The model must have one and only one output "
              "variable. However, it has " +
              utils::stringify(nvar) +
              " output variables. To disambiguite, please specify the variable name using "
              "format 'model_name.variable_name'. The model's output axis is:\n" +
              utils::stringify(provider->output_axis()));

        // Retrieve the output variable
        var_name = provider->output_axis().variable_names()[0];
      }
      // Try to parse it as a variable name
      catch (const FactoryException & err_model)
      {
        auto success = utils::parse_<VariableName>(var_name, tokens[0]);
        if (!success)
          throw ParserException(
              "Invalid variable specifier '" + tn.raw() +
              "'. It should take the form 'model_name', 'variable_name', or "
              "'model_name.variable_name'. Since there is no '.' delimiter, it can either be a "
              "model name or a variable name. Interpreting it as a model name failed with error "
              "message: " +
              err_model.what() + ". It also cannot be parsed as a valid variable name.");

        // Create a dummy model that defines this parameter
        const auto obj_name = "__parameter_" + var_name.str() + "__";
        const auto obj_type = utils::demangle(typeid(T).name()).substr(7) + "InputParameter";
        auto options = InputParameter<T>::expected_options();
        options.template set<std::string>("name") = obj_name;
        options.template set<std::string>("type") = obj_type;
        options.template set<VariableName>("from") = var_name;
        options.template set<VariableName>("to") = var_name.with_suffix("_autogenerated");
        options.set("to").user_specified() = true;
        options.name() = obj_name;
        options.type() = obj_type;
        if (caller->factory()->input_file()["Models"].count(obj_name))
        {
          const auto & existing_options = caller->factory()->input_file()["Models"][obj_name];
          if (!options_compatible(existing_options, options))
            throw ParserException(
                "Option clash when declaring an input parameter. Existing options:\n" +
                utils::stringify(existing_options) + ". New options:\n" +
                utils::stringify(options));
        }
        else
          caller->factory()->input_file()["Models"][obj_name] = std::move(options);

        // Get the model
        provider = caller->factory()->get_object<Model>(
            "Models", obj_name, extra_opts, /*force_create=*/false);

        // Retrieve the output variable
        var_name = provider->output_axis().variable_names()[0];
      }
    }
    else
    {
      // The first token is the model name
      const auto & mname = tokens[0];

      // Get the model
      provider =
          caller->factory()->get_object<Model>("Models", mname, extra_opts, /*force_create=*/false);

      // The second token is the variable name
      auto success = utils::parse_<VariableName>(var_name, tokens[1]);
      if (!success)
        throw ParserException("Invalid variable specifier '" + tn.raw() + "'. '" + tokens[1] +
                              "' cannot be parsed as a valid variable name.");
      if (!provider->output_axis().has_variable(var_name))
        throw ParserException("Invalid variable specifier '" + tn.raw() + "'. Model '" + mname +
                              "' does not have an output variable named '" +
                              utils::stringify(var_name) + "'");
    }

    // Declare the input variable
    caller->declare_input_variable<T>(var_name);

    // Get the variable
    const auto * var = &provider->output_variable(var_name);
    const auto * var_ptr = dynamic_cast<const Variable<T> *>(var);
    if (!var_ptr)
      throw ParserException("The variable specifier '" + tn.raw() +
                            "' is valid, but the variable cannot be cast to type " +
                            utils::demangle(typeid(T).name()));

    // For bookkeeping, the caller shall record the model that provides this variable
    // This is needed for two reasons:
    //   1. When the caller is composed with others, we need this information to automatically
    //      bring in the provider.
    //   2. When the caller is sent to a different device/dtype, the caller needs to forward the
    //      call to the provider.
    caller->register_nonlinear_parameter(pname, NonlinearParameter{provider, var_name, var_ptr});

    // Done!
    return var_ptr->value();
  }
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name,
                                  const TensorName<T> & tensorname,
                                  bool allow_nonlinear)
{
  auto * factory = _object->factory();
  neml_assert(factory, "Internal error: factory != nullptr");

  try
  {
    return declare_parameter(name, tensorname.resolve(factory));
  }
  catch (const SetupException & err_tensor)
  {
    if (allow_nonlinear)
      try
      {
        return resolve_tensor_name(tensorname, _object, name);
      }
      catch (const SetupException & err_var)
      {
        throw ParserException(std::string(err_tensor.what()) +
                              "\nAn additional attempt was made to interpret the tensor name as a "
                              "variable specifier, but it failed with error message:" +
                              err_var.what());
      }
    else
      throw ParserException(
          std::string(err_tensor.what()) +
          "\nThe tensor name cannot be interpreted as a variable specifier because variable "
          "coupling has not been implemented for this parameter. If this is intended, please "
          "consider opening an issue on the NEML2 GitHub repository.");
  }
}

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name,
                                  const std::string & input_option_name,
                                  bool allow_nonlinear)
{
  if (_object->input_options().contains(input_option_name))
    return declare_parameter<T>(
        name, _object->input_options().get<TensorName<T>>(input_option_name), allow_nonlinear);

  throw NEMLException("Trying to register parameter named " + name + " from input option named " +
                      input_option_name + " of type " + utils::demangle(typeid(T).name()) +
                      ". Make sure you provided the correct parameter name, option name, and "
                      "parameter type.");
}

#define PARAMETERSTORE_INTANTIATE_TENSORBASE(T)                                                    \
  template const T & ParameterStore::declare_parameter<T>(const std::string &, const T &)
FOR_ALL_TENSORBASE(PARAMETERSTORE_INTANTIATE_TENSORBASE);

#define PARAMETERSTORE_INTANTIATE_PRIMITIVETENSOR(T)                                               \
  template const T & ParameterStore::declare_parameter<T>(                                         \
      const std::string &, const TensorName<T> &, bool);                                           \
  template const T & ParameterStore::declare_parameter<T>(                                         \
      const std::string &, const std::string &, bool)
FOR_ALL_PRIMITIVETENSOR(PARAMETERSTORE_INTANTIATE_PRIMITIVETENSOR);

void
ParameterStore::assign_parameter_stack(jit::Stack & stack)
{
  const auto & params = _object->host<ParameterStore>()->named_parameters();

  neml_assert_dbg(stack.size() >= params.size(),
                  "Stack size (",
                  stack.size(),
                  ") is smaller than the number of parameters in the model (",
                  params.size(),
                  ").");

  // Last n tensors in the stack are the parameters
  std::size_t i = stack.size() - params.size();
  for (auto && [name, param] : params)
  {
    const auto tensor = stack[i++].toTensor();
    *param = Tensor(tensor, tensor.dim() - Tensor(*param).base_dim());
  }

  // Drop the input variables from the stack
  jit::drop(stack, params.size());
}

jit::Stack
ParameterStore::collect_parameter_stack() const
{
  const auto & params = _object->host<ParameterStore>()->named_parameters();
  jit::Stack stack;
  stack.reserve(params.size());
  for (auto && [name, param] : params)
    stack.emplace_back(Tensor(*param));
  return stack;
}
} // namespace neml2
