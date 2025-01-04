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

#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"
#include "neml2/base/guards.h"
#include "neml2/misc/math.h"
#include "neml2/jit/utils.h"

namespace neml2
{
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

  options.set<bool>("jit") = true;
  options.set("jit").doc() = "Use JIT compilation for the forward operator";

  options.set<bool>("production") = false;
  options.set("production").doc() =
      "Production mode. This option is used to disable features like function graph tracking and "
      "tensor version tracking which are useful for training (i.e., calibrating model parameters) "
      "but are not necessary in production runs.";

  options.set<bool>("_nonlinear_system") = false;
  options.set("_nonlinear_system").suppressed() = true;

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    DiagnosticsInterface(this),
    _nonlinear_system(options.get<bool>("_nonlinear_system")),
    _jit(options.get<bool>("jit")),
    _production(options.get<bool>("production"))
{
}

void
Model::to(const torch::TensorOptions & options)
{
  send_buffers_to(options);
  send_parameters_to(options);
  send_variables_to(options);

  for (auto * submodel : registered_models())
    submodel->to(options);
}

void
Model::diagnose(std::vector<Diagnosis> & diagnoses) const
{
  for (auto * submodel : registered_models())
    submodel->diagnose(diagnoses);

  // Make sure variables are defined on the reserved subaxes
  for (auto && [name, var] : input_variables())
    diagnostic_check_input_variable(diagnoses, var);
  for (auto && [name, var] : output_variables())
    diagnostic_check_output_variable(diagnoses, var);

  if (is_nonlinear_system())
    diagnose_nl_sys(diagnoses);
}

void
Model::diagnose_nl_sys(std::vector<Diagnosis> & diagnoses) const
{
  for (auto * submodel : registered_models())
    submodel->diagnose_nl_sys(diagnoses);

  // Check if any input variable is solve-dependent
  bool input_solve_dep = false;
  for (auto && [name, var] : input_variables())
    if (var.is_solve_dependent())
      input_solve_dep = true;

  // If any input variable is solve-dependent, ALL output variables must be solve-dependent!
  if (input_solve_dep)
    for (auto && [name, var] : output_variables())
      diagnostic_assert(
          diagnoses,
          var.is_solve_dependent(),
          "This model is part of a nonlinear system. At least one of the input variables is "
          "solve-dependent, so all output variables MUST be solve-dependent, i.e., they must be "
          "on one of the following sub-axes: state, residual, parameters. However, got output "
          "variable ",
          name);
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
Model::link_input_variables()
{
  for (auto * submodel : _registered_models)
  {
    link_input_variables(submodel);
    submodel->link_input_variables();
  }
}

void
Model::link_input_variables(Model * submodel)
{
  for (auto && [name, var] : submodel->input_variables())
    var.ref(input_variable(name), submodel->is_nonlinear_system());
}

void
Model::link_output_variables()
{
  for (auto * submodel : _registered_models)
  {
    link_output_variables(submodel);
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
  _ad_derivs[&y].insert(&u);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  _ad_args.insert(const_cast<VariableBase *>(&u));
}

void
Model::request_AD(VariableBase & y, const VariableBase & u1, const VariableBase & u2)
{
  _ad_secderivs[&y][&u1].insert(&u2);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  _ad_args.insert(const_cast<VariableBase *>(&u2));
}

void
Model::clear_input()
{
  VariableStore::clear_input();
  for (auto * submodel : _registered_models)
    submodel->clear_input();
}

void
Model::clear_output()
{
  VariableStore::clear_output();
  for (auto * submodel : _registered_models)
    submodel->clear_output();
}

void
Model::zero_input()
{
  VariableStore::zero_input();
  for (auto * submodel : _registered_models)
    submodel->zero_input();
}

void
Model::zero_output()
{
  VariableStore::zero_output();
  for (auto * submodel : _registered_models)
    submodel->zero_output();
}

Model::TraceSchema
Model::compute_trace_schema() const
{
  std::vector<Size> batch_dims;
  for (auto && [name, var] : input_variables())
    batch_dims.push_back(var.batch_dim());
  for (auto && [name, param] : host<ParameterStore>()->named_parameters())
    batch_dims.push_back(Tensor(param).batch_dim());

  const auto dispatch_key = tensor_options().computeDispatchKey();

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
  torch::InferenceMode mode_guard(_production && !torch::jit::tracer::isTracing());

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
  if (!is_jit_enabled() || torch::jit::tracer::isTracing())
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
    torch::InferenceMode mode_guard(_production);
    auto stack = collect_input_stack();
    traced_function->run(stack);
    assign_output_stack(stack, out, dout, d2out);
  }
  else
  {
    auto forward_wrap = [&](torch::jit::Stack inputs) -> torch::jit::Stack
    {
      assign_input_stack(inputs);
      forward(out, dout, d2out);
      return collect_output_stack(out, dout, d2out);
    };
    auto trace = std::get<0>(torch::jit::tracer::trace(
        collect_input_stack(),
        forward_wrap,
        [this](const torch::Tensor & var) { return variable_name_lookup(var); },
        /*strict=*/false,
        /*force_outplace=*/false));
    auto new_function =
        std::make_unique<torch::jit::GraphFunction>(name() + ".forward",
                                                    trace->graph,
                                                    /*function_creator=*/nullptr,
                                                    torch::jit::ExecutorExecutionMode::PROFILING);
    traced_functions[forward_op_idx].emplace(new_schema, std::move(new_function));

    // Rerun this method -- this time using the jitted graph (without tracing)
    forward_maybe_jit(out, dout, d2out);
  }
}

std::string
Model::variable_name_lookup(const torch::Tensor & var)
{
  // Look for the variable in the input and output variables
  for (auto && [ivar, val] : input_variables())
    if (val.tensor().data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(ivar);
  for (auto && [ovar, val] : output_variables())
    if (val.tensor().data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(ovar);

  // Look for the variable in the parameter and buffer store
  for (auto && [pname, pval] : host<ParameterStore>()->named_parameters())
    if (Tensor(pval).data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(pname);
  for (auto && [bname, bval] : host<BufferStore>()->named_buffers())
    if (Tensor(bval).data_ptr() == var.data_ptr())
      return name() + "::" + utils::stringify(bname);

  // Look for the variable in the registered models
  for (auto * submodel : registered_models())
  {
    const auto name = submodel->variable_name_lookup(var);
    if (!name.empty())
      return name;
  }

  return "";
}

ValueMap
Model::value(const ValueMap & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(true, false, false);

  const auto values = collect_output();
  clear_input();
  clear_output();
  return values;
}

std::tuple<ValueMap, DerivMap>
Model::value_and_dvalue(const ValueMap & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(true, true, false);

  const auto values = collect_output();
  const auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return {values, derivs};
}

DerivMap
Model::dvalue(const ValueMap & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(false, true, false);

  const auto derivs = collect_output_derivatives();
  clear_input();
  clear_output();
  return derivs;
}

std::tuple<ValueMap, DerivMap, SecDerivMap>
Model::value_and_dvalue_and_d2value(const ValueMap & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(true, true, true);

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
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(false, true, true);

  const auto derivs = collect_output_derivatives();
  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return {derivs, secderivs};
}

SecDerivMap
Model::d2value(const ValueMap & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  forward_maybe_jit(false, false, true);

  const auto secderivs = collect_output_second_derivatives();
  clear_input();
  clear_output();
  return secderivs;
}

Model *
Model::registered_model(const std::string & name) const
{
  for (auto * submodel : _registered_models)
    if (submodel->name() == name)
      return submodel;

  throw NEMLException("There is no registered model named '" + name + "' in '" + this->name() +
                      "'");
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
Model::assign_input_stack(torch::jit::Stack & stack)
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

torch::jit::Stack
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

    const auto dy_dus = math::jacrev(y->tensor(),
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

        const auto d2y_du1u2s = math::jacrev(dy_du1,
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
  bool first = false;
  const std::string tab = "            ";

  os << "Name:       " << model.name() << '\n';
  os << "Dtype:      " << model.tensor_options().dtype() << '\n';
  os << "Device:     " << model.tensor_options().device() << '\n';

  if (!model.input_variables().empty())
  {
    os << "Input:      ";
    first = true;
    for (auto && [name, var] : model.input_variables())
    {
      os << (first ? "" : tab);
      os << name << " [" << var.type() << "]\n";
      first = false;
    }
  }

  if (!model.input_variables().empty())
  {
    os << "Output:     ";
    first = true;
    for (auto && [name, var] : model.output_variables())
    {
      os << (first ? "" : tab);
      os << name << " [" << var.type() << "]\n";
      first = false;
    }
  }

  if (!model.named_parameters().empty())
  {
    os << "Parameters: ";
    first = true;
    for (auto && [name, param] : model.named_parameters())
    {
      os << (first ? "" : tab);
      os << name << " [" << param.type() << "]\n";
      first = false;
    }
  }

  if (!model.named_buffers().empty())
  {
    os << "Buffers:    ";
    first = true;
    for (auto && [name, buffer] : model.named_buffers())
    {
      os << (first ? "" : tab);
      os << name << " [" << buffer.type() << "]\n";
      first = false;
    }
  }

  return os;
}
// LCOV_EXCL_STOP
} // namespace neml2
