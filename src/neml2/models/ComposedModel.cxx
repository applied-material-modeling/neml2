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

#include "neml2/models/ComposedModel.h"
#include "neml2/base/Settings.h"

namespace neml2
{
register_NEML2_object(ComposedModel);

OptionSet
ComposedModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Compose multiple models together to form a single model. The composed model can then be "
      "treated as a new model and composed with others. The [system documentation](@ref "
      "model-composition) provides in-depth explanation on how the models are composed together.";

  NonlinearSystem::enable_automatic_scaling(options);

  options.set<std::vector<std::string>>("models");
  options.set("models").doc() = "Models being composed together";

  options.set<std::vector<VariableName>>("additional_outputs");
  options.set("additional_outputs").doc() =
      "Extra output variables to be extracted from the composed model in addition to the ones "
      "identified through dependency resolution.";

  options.set<std::vector<std::string>>("priority");
  options.set("priority").doc() =
      "Priorities of models in decreasing order. A model with higher priority will be evaluated "
      "first. This is useful for breaking cyclic dependency.";

  options.set<bool>("automatic_nonlinear_parameter") = true;
  options.set("automatic_nonlinear_parameter").doc() =
      "Whether to automatically add dependent nonlinear parameters";

  return options;
}

ComposedModel::ComposedModel(const OptionSet & options)
  : Model(options),
    _additional_outputs(options.get<std::vector<VariableName>>("additional_outputs")),
    _auto_nl_param(options.get<bool>("automatic_nonlinear_parameter"))
{
  // Each sub-model shall have _independent_ output storage. This is because the same model could
  // be registered as a sub-model by different models, and it could be evaluated with _different_
  // input, and hence yields _different_ output values.
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
    register_model(model_name, /*nonlinear=*/false, /*merge_input=*/false);

  // Each sub-model may have nonlinear parameters. In our design, nonlinear parameters _are_
  // models. Since we do not want to put the burden of "adding nonlinear parameters in the input
  // file through the option 'models'" on users, we should do more behind the scenes to register
  // them.
  //
  // Registering nonlinear parameters here ensures dependency resolution. And if a nonlinear
  // parameter is registered by multiple models (which is very possible), we won't have to
  // evaluate the nonlinar parameter over and over again!
  auto submodels = registered_models();
  if (_auto_nl_param)
    for (auto & submodel : submodels)
      for (auto && [pname, param] : submodel->named_nonlinear_parameters(/*recursive=*/true))
        if (std::find(_registered_models.begin(), _registered_models.end(), param.provider) ==
            _registered_models.end())
          _registered_models.push_back(param.provider);

  // Add registered models as nodes in the dependency resolver
  for (auto & submodel : registered_models())
    _dependency.add_node(submodel.get());
  for (const auto & var : _additional_outputs)
    _dependency.add_additional_outbound_item(var);

  // Define priority in the event of cyclic dependency
  auto priority_order = options.get<std::vector<std::string>>("priority");
  size_t priority = priority_order.empty() ? 0 : priority_order.size() - 1;
  for (const auto & model_name : priority_order)
    _dependency.set_priority(registered_model(model_name).get(), priority--);

  // Resolve the dependency
  _dependency.unique_item_provider() = true;
  _dependency.unique_item_consumer() = false;
  _dependency.resolve();
  const auto & resolution = _dependency.resolution();

  // Sort the registered models by dependency resolution
  std::vector<std::shared_ptr<Model>> sorted_models(resolution.size());
  for (std::size_t i = 0; i < resolution.size(); ++i)
    sorted_models[i] = registered_model(resolution[i]->name());
  _registered_models = std::move(sorted_models);

  // Register input variables
  for (const auto & item : _dependency.inbound_items())
    if (!input_variables().count(item.value))
      clone_input_variable(item.parent->input_variable(item.value));

  // Register output variables
  for (const auto & item : _dependency.outbound_items())
    clone_output_variable(item.parent->output_variable(item.value));

  // Declare nonlinear parameters
  for (auto & submodel : submodels)
    for (auto && [pname, param] : submodel->named_nonlinear_parameters(/*recursive=*/true))
      if (input_variables().count(param.provider_var))
        register_nonlinear_parameter(pname, param);

  // Check if this composed model defines values
  _defines_value = true;
  for (const auto & submodel : registered_models())
    if (!submodel->defines_values())
    {
      _defines_value = false;
      _defines_dvalue = false;
      _defines_d2value = false;
      break;
    }

  // Check if this composed model defines derivatives
  if (_defines_value)
  {
    _defines_dvalue = true;
    for (const auto & submodel : registered_models())
      if (!submodel->defines_derivatives())
      {
        _defines_dvalue = false;
        _defines_d2value = false;
        break;
      }
  }

  // Check if this composed model defines second derivatives
  if (_defines_dvalue)
  {
    _defines_d2value = true;
    for (const auto & submodel : registered_models())
      if (!submodel->defines_second_derivatives())
      {
        _defines_d2value = false;
        break;
      }
  }

  // Is JIT enabled?
  _jit = Model::is_jit_enabled(); // boolean read from input file

  // If any submodel does not support JIT, disable JIT
  if (_jit)
    for (const auto & submodel : registered_models())
      if (!submodel->is_jit_enabled())
      {
        _jit = false;
        break;
      }
}

std::map<std::string, NonlinearParameter>
ComposedModel::named_nonlinear_parameters(bool /*recursive*/) const
{
  return _nl_params;
}

void
ComposedModel::link_input_variables(Model * submodel)
{
  for (const auto & item : _dependency.inbound_items())
    if (item.parent == submodel)
      submodel->input_variable(item.value).ref(input_variable(item.value));

  for (const auto & [item, providers] : _dependency.item_providers())
    if (item.parent == submodel)
    {
      auto * depmodel = providers.begin()->parent;
      submodel->input_variable(item.value).ref(depmodel->output_variable(item.value));
    }
}

void
ComposedModel::link_output_variables(Model * submodel)
{
  for (const auto & item : _dependency.outbound_items())
    if (item.parent == submodel)
      output_variable(item.value).ref(submodel->output_variable(item.value));
}

void
ComposedModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  for (const auto & i : registered_models())
  {
    if (out && !dout_din && !d2out_din2)
      i->forward_maybe_jit(true, false, false);
    else if (dout_din && !d2out_din2)
      i->forward_maybe_jit(true, true, false);
    else if (d2out_din2)
      i->forward_maybe_jit(true, true, true);
    else
      throw NEMLException("Unsupported call signature to set_value");
  }

  if (dout_din)
    for (auto && [name, var] : output_variables())
      var->apply_chain_rule(_dependency);

  if (d2out_din2)
    for (auto && [name, var] : output_variables())
      var->apply_second_order_chain_rule(_dependency);
}
} // namespace neml2
