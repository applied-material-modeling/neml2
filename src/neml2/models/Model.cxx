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

#include <c10/core/InferenceMode.h>

#include "neml2/misc/assertions.h"
#include "neml2/base/guards.h"
#include "neml2/base/Factory.h"
#include "neml2/base/Settings.h"
#include "neml2/jit/utils.h"
#include "neml2/tensors/functions/jacrev.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/TensorValue.h"
#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"
#include "neml2/models/map_types_fwd.h"
#include "neml2/models/BundledModel.h"

namespace neml2
{
std::shared_ptr<Model>
load_model(const std::filesystem::path & path, const std::string & mname)
{
  auto factory = load_input(path);
  return factory->get_model(mname);
}

bool
Model::TraceSchema::operator==(const TraceSchema & other) const
{
  return batch_dims == other.batch_dims && dispatch_key == other.dispatch_key;
}

bool
Model::TraceSchema::operator<(const TraceSchema & other) const
{
  if (dispatch_key != other.dispatch_key)
    return dispatch_key < other.dispatch_key;
  return batch_dims < other.batch_dims;
}

OptionSet
Model::expected_options()
{
  OptionSet options = Data::expected_options();
  options += NonlinearSystem::expected_options();
  NonlinearSystem::disable_automatic_scaling(options);

  options.section() = "Models";

  // Model defaults to defining value and dvalue, but not d2value
  options.set<bool>("define_values") = true;
  options.set<bool>("define_derivatives") = true;
  options.set<bool>("define_second_derivatives") = false;
  options.set("define_values").suppressed() = true;
  options.set("define_derivatives").suppressed() = true;
  options.set("define_second_derivatives").suppressed() = true;

  // Model defaults to _not_ being part of a nonlinear system
  // Model::get_model will set this to true if the model is expected to be part of a nonlinear
  // system, and additional diagnostics will be performed
  options.set<bool>("nonlinear_system") = false;
  options.set("nonlinear_system").suppressed() = true;

  options.set<bool>("jit") = true;
  options.set("jit").doc() = "Use JIT compilation for the forward operator";

  options.set<bool>("production") = false;
  options.set("production").doc() =
      "Production mode. This option is used to disable features like function graph tracking and "
      "tensor version tracking which are useful for training (i.e., calibrating model parameters) "
      "but are not necessary in production runs.";

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(this),
    VariableStore(this),
    NonlinearSystem(options),
    DiagnosticsInterface(this),
    _defines_value(options.get<bool>("define_values")),
    _defines_dvalue(options.get<bool>("define_derivatives")),
    _defines_d2value(options.get<bool>("define_second_derivatives")),
    _nonlinear_system(options.get<bool>("nonlinear_system")),
    _jit(options.get<bool>("jit")),
    _production(options.get<bool>("production"))
{
}

void
Model::to(const TensorOptions & options)
{
  send_buffers_to(options);
  send_parameters_to(options);
  send_variables_to(options);

  for (auto & submodel : registered_models())
    submodel->to(options);

  for (auto & [name, param] : named_nonlinear_parameters())
    param.provider->to(options);
}

void
Model::setup()
{
  setup_layout();

  if (host() == this)
  {
    link_output_variables();
    link_input_variables();
  }

  request_AD();
}

void
Model::diagnose() const
{
  for (auto & submodel : registered_models())
    neml2::diagnose(*submodel);

  // Make sure variables are defined on the reserved subaxes
  for (auto && [name, var] : input_variables())
    diagnostic_check_input_variable(*var);
  for (auto && [name, var] : output_variables())
    diagnostic_check_output_variable(*var);

  if (is_nonlinear_system())
    diagnose_nl_sys();
}

void
Model::diagnose_nl_sys() const
{
  for (auto & submodel : registered_models())
    submodel->diagnose_nl_sys();

  // Check if any input variable is solve-dependent
  bool input_solve_dep = false;
  for (auto && [name, var] : input_variables())
    if (var->is_solve_dependent())
      input_solve_dep = true;

  // If any input variable is solve-dependent, ALL output variables must be solve-dependent!
  if (input_solve_dep)
    for (auto && [name, var] : output_variables())
      diagnostic_assert(
          var->is_solve_dependent(),
          "This model is part of a nonlinear system. At least one of the input variables is "
          "solve-dependent, so all output variables MUST be solve-dependent, i.e., they must be "
          "on one of the following sub-axes: state, residual, parameters. However, got output "
          "variable ",
          name);
}

void
Model::diagnostic_assert_state(const VariableBase & v) const
{
  diagnostic_assert(v.is_state(), "Variable ", v.name(), " must be on the ", STATE, " sub-axis.");
}

void
Model::diagnostic_assert_old_state(const VariableBase & v) const
{
  diagnostic_assert(
      v.is_old_state(), "Variable ", v.name(), " must be on the ", OLD_STATE, " sub-axis.");
}

void
Model::diagnostic_assert_force(const VariableBase & v) const
{
  diagnostic_assert(v.is_force(), "Variable ", v.name(), " must be on the ", FORCES, " sub-axis.");
}

void
Model::diagnostic_assert_old_force(const VariableBase & v) const
{
  diagnostic_assert(
      v.is_old_force(), "Variable ", v.name(), " must be on the ", OLD_FORCES, " sub-axis.");
}

void
Model::diagnostic_assert_residual(const VariableBase & v) const
{
  diagnostic_assert(
      v.is_residual(), "Variable ", v.name(), " must be on the ", RESIDUAL, " sub-axis.");
}

void
Model::diagnostic_check_input_variable(const VariableBase & v) const
{
  diagnostic_assert(v.is_state() || v.is_old_state() || v.is_force() || v.is_old_force() ||
                        v.is_residual() || v.is_parameter(),
                    "Input variable ",
                    v.name(),
                    " must be on one of the following sub-axes: ",
                    STATE,
                    ", ",
                    OLD_STATE,
                    ", ",
                    FORCES,
                    ", ",
                    OLD_FORCES,
                    ", ",
                    RESIDUAL,
                    ", ",
                    PARAMETERS,
                    ".");
}

void
Model::diagnostic_check_output_variable(const VariableBase & v) const
{
  diagnostic_assert(v.is_state() || v.is_force() || v.is_residual() || v.is_parameter(),
                    "Output variable ",
                    v.name(),
                    " must be on one of the following sub-axes: ",
                    STATE,
                    ", ",
                    FORCES,
                    ", ",
                    RESIDUAL,
                    ", ",
                    PARAMETERS,
                    ".");
}

void
Model::link_input_variables()
{
  for (auto & submodel : _registered_models)
  {
    link_input_variables(submodel.get());
    submodel->link_input_variables();
  }
}

void
Model::link_input_variables(Model * submodel)
{
  for (auto && [name, var] : submodel->input_variables())
    var->ref(input_variable(name), submodel->is_nonlinear_system());
}

void
Model::link_output_variables()
{
  for (auto & submodel : _registered_models)
  {
    link_output_variables(submodel.get());
    submodel->link_output_variables();
  }
}

void
Model::link_output_variables(Model * /*submodel*/)
{
}

void
Model::request_AD(VariableBase & y, const VariableBase & u)
{
  neml_assert(_defines_value,
              "Model of type '",
              type(),
              "' is requesting automatic differentiation of first derivatives, but it does not "
              "define output values.");
  _defines_dvalue = true;
  _ad_derivs[&y].insert(&u);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  _ad_args.insert(const_cast<VariableBase *>(&u));
}

void
Model::request_AD(VariableBase & y, const VariableBase & u1, const VariableBase & u2)
{
  neml_assert(_defines_dvalue,
              "Model of type '",
              type(),
              "' is requesting automatic differentiation of second derivatives, but it does not "
              "define first derivatives.");
  _defines_d2value = true;
  _ad_secderivs[&y][&u1].insert(&u2);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  _ad_args.insert(const_cast<VariableBase *>(&u2));
}

void
Model::clear_input()
{
  VariableStore::clear_input();
  for (auto & submodel : _registered_models)
    submodel->clear_input();
}

void
Model::clear_output()
{
  VariableStore::clear_output();
  for (auto & submodel : _registered_models)
    submodel->clear_output();
}

void
Model::zero_input()
{
  VariableStore::zero_input();
  for (auto & submodel : _registered_models)
    submodel->zero_input();
}

void
Model::zero_output()
{
  VariableStore::zero_output();
  for (auto & submodel : _registered_models)
    submodel->zero_output();
}

Model::TraceSchema
Model::compute_trace_schema() const
{
  std::vector<Size> batch_dims;
  for (auto && [name, var] : input_variables())
    batch_dims.push_back(var->batch_dim());
  for (auto && [name, param] : host<ParameterStore>()->named_parameters())
    batch_dims.push_back(Tensor(*param).batch_dim());

  const auto dispatch_key = variable_options().computeDispatchKey();

  return TraceSchema{batch_dims, dispatch_key};
}

std::size_t
Model::forward_operator_index(bool out, bool dout, bool d2out) const
{
  return (out ? 4 : 0) + (dout ? 2 : 0) + (d2out ? 1 : 0);
}

void
Model::forward(bool out, bool dout, bool d2out)
{
  neml_assert_dbg(defines_values() || (defines_values() == out),
                  "Model of type '",
                  type(),
                  "' is requested to compute output values, but it does not define them.");
  neml_assert_dbg(defines_derivatives() || (defines_derivatives() == dout),
                  "Model of type '",
                  type(),
                  "' is requested to compute first derivatives, but it does not define them.");
  neml_assert_dbg(defines_second_derivatives() || (defines_second_derivatives() == d2out),
                  "Model of type '",
                  type(),
                  "' is requested to compute second derivatives, but it does not define them.");

  c10::InferenceMode mode_guard(_production && !jit::tracer::isTracing());

  if (dout || d2out)
    enable_AD();

  set_value(out || AD_need_value(dout, d2out), dout, d2out);

  if (dout || d2out)
    extract_AD_derivatives(dout, d2out);

  return;
}

void
Model::forward_maybe_jit(bool out, bool dout, bool d2out)
{
  if (!is_jit_enabled() || jit::tracer::isTracing())
  {
    forward(out, dout, d2out);
    return;
  }

  auto & traced_functions =
      currently_solving_nonlinear_system() ? _traced_functions_nl_sys : _traced_functions;

  const auto forward_op_idx = forward_operator_index(out, dout, d2out);
  const auto new_schema = compute_trace_schema();
  auto traced_schema_and_function = traced_functions[forward_op_idx].find(new_schema);

  if (traced_schema_and_function != traced_functions[forward_op_idx].end())
  {
    auto & [trace_schema, traced_function] = *traced_schema_and_function;
    c10::InferenceMode mode_guard(_production);
    auto stack = collect_input_stack();
    traced_function->run(stack);
    assign_output_stack(stack, out, dout, d2out);
  }
  else
  {
    // All other models in the world should wait for this model to finish tracing
    // This is not our fault, torch jit tracing is not thread-safe
    static std::mutex trace_mutex;
    trace_mutex.lock();
    auto forward_wrap = [&](jit::Stack inputs) -> jit::Stack
    {
      assign_input_stack(inputs);
      forward(out, dout, d2out);
      return collect_output_stack(out, dout, d2out);
    };
    auto trace = std::get<0>(jit::tracer::trace(
        collect_input_stack(),
        forward_wrap,
        [this](const ATensor & var) { return variable_name_lookup(var); },
        /*strict=*/false,
        /*force_outplace=*/false));
    trace_mutex.unlock();

    auto new_function = std::make_unique<jit::GraphFunction>(name() + ".forward",
                                                             trace->graph,
                                                             /*function_creator=*/nullptr,
                                                             jit::ExecutorExecutionMode::PROFILING);
    traced_functions[forward_op_idx].emplace(new_schema, std::move(new_function));

    // Rerun this method -- this time using the jitted graph (without tracing)
    forward_maybe_jit(out, dout, d2out);
  }
}

std::string
Model::variable_name_lookup(const ATensor & var)
{
  // Look for the variable in the input and output variables
  for (auto && [ivar, val] : input_variables())
    if (val->tensor().data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(ivar);
  for (auto && [ovar, val] : output_variables())
    if (val->tensor().data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(ovar);

  // Look for the variable in the parameter and buffer store
  for (auto && [pname, pval] : host<ParameterStore>()->named_parameters())
    if (Tensor(*pval).data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(pname);
  for (auto && [bname, bval] : host<BufferStore>()->named_buffers())
    if (Tensor(*bval).data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(bname);

  // Look for the variable in the registered models
  for (auto & submodel : registered_models())
  {
    auto name = submodel->variable_name_lookup(var);
    if (!name.empty())
      return name;
  }

  return "";
}

void
Model::check_precision() const
{
  if (settings().require_double_precision())
    neml_assert(
        default_tensor_options().dtype() == kFloat64,
        "By default, NEML2 requires double precision for all computations. Please set the default "
        "dtype to Float64. In Python, this can be done by calling "
        "`torch.set_default_dtype(torch.double)`. In C++, this can be done by calling "
        "`neml2::set_default_dtype(neml2::kFloat64)`. If other precisions are truly needed, you "
        "can disable this error check with Settings/require_double_precision=false.");
}

ValueMap
Model::value(const ValueMap & in)
{
  forward_helper(in, true, false, false);

  auto values = collect_output();
  clear_input();
  clear_output();
  return values;
}

ValueMap
Model::value(ValueMap && in)
{
  forward_helper(std::move(in), true, false, false);

  auto values = collect_output();
  clear_input();
  clear_output();
  return values;
}

std::tuple<ValueMap, DerivMap>
Model::value_and_dvalue(const ValueMap & in)
{
  forward_helper(in, true, true, false);

  const auto values = collect_output();
  const auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return {values, derivs};
}

std::tuple<ValueMap, DerivMap>
Model::value_and_dvalue(ValueMap && in)
{
  forward_helper(std::move(in), true, true, false);

  const auto values = collect_output();
  const auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return {values, derivs};
}

DerivMap
Model::dvalue(const ValueMap & in)
{
  forward_helper(in, false, true, false);

  auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return derivs;
}

DerivMap
Model::dvalue(ValueMap && in)
{
  forward_helper(std::move(in), false, true, false);

  auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return derivs;
}

std::tuple<ValueMap, DerivMap, SecDerivMap>
Model::value_and_dvalue_and_d2value(const ValueMap & in)
{
  forward_helper(in, true, true, true);

  const auto values = collect_output();
  const auto derivs = collect_output_derivatives();
  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return {values, derivs, secderivs};
}

std::tuple<ValueMap, DerivMap, SecDerivMap>
Model::value_and_dvalue_and_d2value(ValueMap && in)
{
  forward_helper(std::move(in), true, true, true);

  const auto values = collect_output();
  const auto derivs = collect_output_derivatives();
  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return {values, derivs, secderivs};
}

std::tuple<DerivMap, SecDerivMap>
Model::dvalue_and_d2value(const ValueMap & in)
{
  forward_helper(in, false, true, true);

  const auto derivs = collect_output_derivatives();
  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return {derivs, secderivs};
}

std::tuple<DerivMap, SecDerivMap>
Model::dvalue_and_d2value(ValueMap && in)
{
  forward_helper(std::move(in), false, true, true);

  const auto derivs = collect_output_derivatives();
  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return {derivs, secderivs};
}

SecDerivMap
Model::d2value(const ValueMap & in)
{
  forward_helper(in, false, false, true);

  auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return secderivs;
}

SecDerivMap
Model::d2value(ValueMap && in)
{
  forward_helper(std::move(in), false, false, true);

  auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return secderivs;
}

std::shared_ptr<Model>
Model::registered_model(const std::string & name) const
{
  for (auto & submodel : _registered_models)
    if (submodel->name() == name)
      return submodel;

  throw NEMLException("There is no registered model named '" + name + "' in '" + this->name() +
                      "'");
}

void
Model::register_nonlinear_parameter(const std::string & pname, const NonlinearParameter & param)
{
  neml_assert(_nl_params.count(pname) == 0,
              "Nonlinear parameter named '",
              pname,
              "' has already been registered.");
  _nl_params[pname] = param;
}

bool
Model::has_nl_param(bool recursive) const
{
  if (!recursive)
    return !_nl_params.empty();

  for (auto & submodel : registered_models())
    if (submodel->has_nl_param(true))
      return true;

  return false;
}

const VariableBase *
Model::nl_param(const std::string & name) const
{
  return _nl_params.count(name) ? _nl_params.at(name).value : nullptr;
}

std::map<std::string, NonlinearParameter>
Model::named_nonlinear_parameters(bool recursive) const
{
  if (!recursive)
    return _nl_params;

  auto all_nl_params = _nl_params;

  for (const auto & [pname, param] : _nl_params)
    for (auto && [pname, nl_param] : param.provider->named_nonlinear_parameters(true))
      all_nl_params[param.provider->name() + settings().parameter_name_separator() + pname] =
          nl_param;

  for (auto & submodel : registered_models())
    for (auto && [pname, nl_param] : submodel->named_nonlinear_parameters(true))
      all_nl_params[submodel->name() + settings().parameter_name_separator() + pname] = nl_param;

  return all_nl_params;
}

std::set<VariableName>
Model::consumed_items() const
{
  auto items = input_axis().variable_names();
  return {items.begin(), items.end()};
}

std::set<VariableName>
Model::provided_items() const
{
  auto items = output_axis().variable_names();
  return {items.begin(), items.end()};
}

void
Model::assign_input_stack(jit::Stack & stack)
{
#ifndef NDEBUG
  const auto nstack = input_axis().nvariable() + host<ParameterStore>()->named_parameters().size();
  neml_assert_dbg(
      stack.size() == nstack,
      "Stack size (",
      stack.size(),
      ") must equal to the number of input variables, parameters, and buffers in the model (",
      nstack,
      ").");
#endif

  assign_parameter_stack(stack);
  VariableStore::assign_input_stack(stack);
}

jit::Stack
Model::collect_input_stack() const
{
  auto stack = VariableStore::collect_input_stack();
  const auto param_stack = collect_parameter_stack();

  // Recall stack is first in last out.
  // Parameter stack go after (on top of) input variables. This means that in assign_input_stack
  // we need to pop parameters first, then input variables.
  stack.insert(stack.end(), param_stack.begin(), param_stack.end());
  return stack;
}

void
Model::set_guess(const Sol<false> & x)
{
  const auto sol_assember = VectorAssembler(input_axis().subaxis(STATE));
  assign_input(sol_assember.split_by_variable(x));
}

void
Model::assemble(NonlinearSystem::Res<false> * residual, NonlinearSystem::Jac<false> * Jacobian)
{
  forward_maybe_jit(residual, Jacobian, false);

  if (residual)
  {
    const auto res_assembler = VectorAssembler(output_axis().subaxis(RESIDUAL));
    *residual = Res<false>(res_assembler.assemble_by_variable(collect_output()));
  }
  if (Jacobian)
  {
    const auto jac_assembler =
        MatrixAssembler(output_axis().subaxis(RESIDUAL), input_axis().subaxis(STATE));
    *Jacobian = Jac<false>(jac_assembler.assemble_by_variable(collect_output_derivatives()));
  }
}

bool
Model::AD_need_value(bool dout, bool d2out) const
{
  if (dout)
    if (!_ad_derivs.empty())
      return true;

  if (d2out)
    for (auto && [y, u1u2s] : _ad_secderivs)
      for (auto && [u1, u2s] : u1u2s)
        if (_ad_derivs.count(y) && _ad_derivs.at(y).count(u1))
          return true;

  return false;
}

void
Model::enable_AD()
{
  for (auto * ad_arg : _ad_args)
    ad_arg->requires_grad_();
}

void
Model::extract_AD_derivatives(bool dout, bool d2out)
{
  neml_assert(dout || d2out, "At least one of the output derivatives must be requested.");

  for (auto && [y, us] : _ad_derivs)
  {
    if (!dout && d2out)
      if (!_ad_secderivs.count(y))
        continue;

    // Gather all dependent variables
    std::vector<Tensor> uts;
    for (const auto * u : us)
      if (u->is_dependent())
        uts.push_back(u->tensor());

    // Check if we need to create the graph (i.e., if any of the second derivatives are requested)
    bool create_graph = false;
    for (const auto * u : us)
      if (u->is_dependent())
        if (!create_graph && !dout && d2out)
          if (_ad_secderivs.at(y).count(u))
            create_graph = true;

    const auto dy_dus = jacrev(y->tensor(),
                               uts,
                               /*retain_graph=*/true,
                               /*create_graph=*/create_graph,
                               /*allow_unused=*/true);

    std::size_t i = 0;
    for (const auto * u : us)
      if (u->is_dependent())
      {
        if (dy_dus[i].defined())
          y->d(*u) = dy_dus[i];
        i++;
      }
  }

  if (d2out)
  {
    for (auto && [y, u1u2s] : _ad_secderivs)
      for (auto && [u1, u2s] : u1u2s)
      {
        if (!u1->is_dependent())
          continue;

        const auto & dy_du1 = y->derivatives()[u1->name()];

        if (!dy_du1.defined() || !dy_du1.requires_grad())
          continue;

        std::vector<Tensor> u2ts;
        for (const auto * u2 : u2s)
          if (u2->is_dependent())
            u2ts.push_back(u2->tensor());

        const auto d2y_du1u2s = jacrev(dy_du1,
                                       u2ts,
                                       /*retain_graph=*/true,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);

        std::size_t i = 0;
        for (const auto * u2 : u2s)
          if (u2->is_dependent())
          {
            if (d2y_du1u2s[i].defined())
              y->d(*u1, *u2) = d2y_du1u2s[i];
            i++;
          }
      }
  }
}

// LCOV_EXCL_START
std::ostream &
operator<<(std::ostream & os, const Model & model)
{
  const auto * bundled_model = dynamic_cast<const BundledModel *>(&model);
  bool first = false;
  const std::string tab = "             ";

  os << "Name:        " << model.name() << '\n';
  if (bundled_model)
    os << "Description: " << bundled_model->description() << '\n';

  if (!model.input_variables().empty())
  {
    os << "Input:       ";
    first = true;
    for (auto && [name, var] : model.input_variables())
    {
      os << (first ? "" : tab);
      os << name << " [" << var->type() << "]";
      if (bundled_model)
        os << " (" << bundled_model->input_description(name) << ")";
      os << '\n';
      first = false;
    }
  }

  if (!model.output_variables().empty())
  {
    os << "Output:      ";
    first = true;
    for (auto && [name, var] : model.output_variables())
    {
      os << (first ? "" : tab);
      os << name << " [" << var->type() << "]";
      if (bundled_model)
        os << " (" << bundled_model->output_description(name) << ")";
      os << '\n';
      first = false;
    }
  }

  if (!model.named_parameters().empty())
  {
    os << "Parameters:  ";
    first = true;
    for (auto && [name, param] : model.named_parameters())
    {
      os << (first ? "" : tab);
      os << name << " [" << param->type() << "][" << Tensor(*param).scalar_type() << "]["
         << Tensor(*param).device() << "]";
      if (bundled_model)
        os << " (" << bundled_model->param_description(name) << ")";
      os << '\n';
      first = false;
    }
  }

  if (!model.named_buffers().empty())
  {
    os << "Buffers:     ";
    first = true;
    for (auto && [name, buffer] : model.named_buffers())
    {
      os << (first ? "" : tab);
      os << name << " [" << buffer->type() << "][" << Tensor(*buffer).scalar_type() << "]["
         << Tensor(*buffer).device() << "]";
      if (bundled_model)
        os << " (" << bundled_model->buffer_description(name) << ")";
      os << '\n';
      first = false;
    }
  }

  return os;
}
// LCOV_EXCL_STOP
} // namespace neml2
