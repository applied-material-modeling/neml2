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

#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"
#include "neml2/base/guards.h"
#include "neml2/misc/math.h"

namespace neml2
{
OptionSet
Model::expected_options()
{
  OptionSet options = Data::expected_options();
  options += NonlinearSystem::expected_options();
  NonlinearSystem::disable_automatic_scaling(options);

  options.section() = "Models";

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
    _nonlinear_system(options.get<bool>("_nonlinear_system"))
{
}

void
Model::to(const torch::TensorOptions & options)
{
  send_buffers_to(options);
  send_parameters_to(options);

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

  // Check for statefulness
  if (this == host())
    if (input_axis().has_old_state())
      for (auto var : input_axis().subaxis("old_state").variable_names())
        diagnostic_assert(diagnoses,
                          output_axis().has_variable(var.prepend("state")),
                          "Input axis has old state variable ",
                          var,
                          ", but the corresponding output state variable doesn't exist.");
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

  {
    RequestingAD AD;
    request_AD();
  }
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

std::map<VariableName, Tensor>
Model::value(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  value();
  return collect_output();
}

std::tuple<std::map<VariableName, Tensor>, std::map<VariableName, std::map<VariableName, Tensor>>>
Model::value_and_dvalue(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  value_and_dvalue();
  return {collect_output(), collect_output_derivatives()};
}

std::map<VariableName, std::map<VariableName, Tensor>>
Model::dvalue(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  dvalue();
  return collect_output_derivatives();
}

std::tuple<std::map<VariableName, Tensor>,
           std::map<VariableName, std::map<VariableName, Tensor>>,
           std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>>
Model::value_and_dvalue_and_d2value(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  value_and_dvalue_and_d2value();
  return {collect_output(), collect_output_derivatives(), collect_output_second_derivatives()};
}

std::tuple<std::map<VariableName, std::map<VariableName, Tensor>>,
           std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>>
Model::dvalue_and_d2value(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  dvalue_and_d2value();
  return {collect_output_derivatives(), collect_output_second_derivatives()};
}

std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>
Model::d2value(const std::map<VariableName, Tensor> & in)
{
  zero_input();
  assign_input(in);
  zero_output();
  d2value();
  return collect_output_second_derivatives();
}

void
Model::value()
{
  set_value(true, false, false);
}

void
Model::value_and_dvalue()
{
  enable_AD();
  set_value(true, true, false);
  extract_AD_derivatives(true, false);
}

void
Model::dvalue()
{
  enable_AD();
  set_value(AD_need_value(true, false), true, false);
  extract_AD_derivatives(true, false);
}

void
Model::value_and_dvalue_and_d2value()
{
  enable_AD();
  set_value(true, true, true);
  extract_AD_derivatives(true, true);
}

void
Model::dvalue_and_d2value()
{
  enable_AD();
  set_value(AD_need_value(true, true), true, true);
  extract_AD_derivatives(true, true);
}

void
Model::d2value()
{
  enable_AD();
  set_value(AD_need_value(false, true), false, true);
  extract_AD_derivatives(false, true);
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
Model::set_guess(const SOL<false> & x)
{
  const auto sol_assember = VectorAssembler(input_axis().subaxis("state"));
  assign_input(sol_assember.disassemble(x));
}

void
Model::assemble(NonlinearSystem::RES<false> * residual, NonlinearSystem::JAC<false> * Jacobian)
{
  if (residual && !Jacobian)
    value();
  else if (!residual && Jacobian)
    dvalue();
  else if (residual && Jacobian)
    value_and_dvalue();

  if (residual)
  {
    const auto res_assembler = VectorAssembler(output_axis().subaxis("residual"));
    *residual = RES<false>(res_assembler.assemble(collect_output()));
  }
  if (Jacobian)
  {
    const auto jac_assembler =
        MatrixAssembler(output_axis().subaxis("residual"), input_axis().subaxis("state"));
    *Jacobian = JAC<false>(jac_assembler.assemble(collect_output_derivatives()));
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

  bool create_graph = false;

  for (auto && [y, us] : _ad_derivs)
  {
    if (!dout && d2out)
      if (!_ad_secderivs.count(y))
        continue;

    for (const auto * u : us)
    {
      if (!dout && d2out)
      {
        if (!_ad_secderivs.at(y).count(u))
          continue;
        create_graph = true;
      }

      const auto dy_du = math::jacrev(y->tensor(),
                                      u->tensor(),
                                      /*retain_graph=*/true,
                                      /*create_graph=*/create_graph,
                                      /*allow_unused=*/true);
      if (dy_du.defined())
        y->d(*u) = dy_du;
    }
  }

  if (d2out)
  {
    for (auto && [y, u1u2s] : _ad_secderivs)
      for (auto && [u1, u2s] : u1u2s)
      {
        const auto & dy_du1 = y->derivatives()[u1->name()];

        if (!dy_du1.defined() || !dy_du1.requires_grad())
          continue;

        for (const auto * u2 : u2s)
        {
          const auto d2y_du1u2 = math::jacrev(dy_du1,
                                              u2->tensor(),
                                              /*retain_graph=*/true,
                                              /*create_graph=*/false,
                                              /*allow_unused=*/true);
          if (d2y_du1u2.defined())
            y->d(*u1, *u2) = d2y_du1u2;
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
