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

#include "neml2/models/ModelNonlinearSystem.h"
#include "neml2/equation_systems/EquationSystem.h"
#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/misc/errors.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"

#include <set>

namespace neml2
{
template <typename VariableMap, typename Predicate>
static std::vector<std::vector<LabeledAxisAccessor>>
build_grouped_map(const VariableMap & vars,
                  const std::optional<std::vector<std::vector<VariableName>>> & requested_groups,
                  Predicate predicate,
                  const std::string & group_option,
                  const std::string & variable_kind)
{
  if (!requested_groups.has_value())
  {
    std::vector<LabeledAxisAccessor> vars_in_one_group;
    for (const auto & [vname, var] : vars)
      if (predicate(*var))
        vars_in_one_group.push_back(vname);
    return {std::move(vars_in_one_group)};
  }

  const auto & groups = requested_groups.value();
  if (groups.empty())
    throw NEMLException("Option '" + group_option + "' must contain at least one group.");

  std::set<std::string> model_vars;
  for (const auto & [vname, var] : vars)
    if (predicate(*var))
      model_vars.insert(vname.str());

  std::set<std::string> seen;
  for (const auto & group : groups)
    for (const auto & vname : group)
    {
      if (!model_vars.count(vname.str()))
        throw NEMLException("Variable '" + vname.str() + "' in '" + group_option + "' is not a " +
                            variable_kind + " variable.");
      if (!seen.insert(vname.str()).second)
        throw NEMLException(variable_kind + " variable '" + vname.str() +
                            "' appears in multiple entries of '" + group_option + "'.");
    }

  for (const auto & v : model_vars)
    if (!seen.count(v))
      throw NEMLException(variable_kind + " variable '" + v + "' is not included in '" +
                          group_option + "'.");

  std::vector<std::vector<LabeledAxisAccessor>> mapped_groups;
  for (const auto & group : groups)
  {
    std::vector<LabeledAxisAccessor> mapped_group;
    mapped_group.reserve(group.size());
    for (const auto & vname : group)
      mapped_group.push_back(vname);
    mapped_groups.push_back(std::move(mapped_group));
  }

  return mapped_groups;
}

register_NEML2_object_alias(ModelNonlinearSystem, "NonlinearSystem");

OptionSet
ModelNonlinearSystem::expected_options()
{
  OptionSet options = EquationSystem::expected_options();

  options.doc() = "A nonlinear system defined by a Model.";

  options.set<std::string>("model") = "model";
  options.set("model").doc() = "The Model defining this nonlinear system.";

  options.set<std::vector<std::vector<VariableName>>>("unknown_groups");
  options.set("unknown_groups").doc() = "Optional grouping of unknown/state variables. "
                                        "Each inner list defines one variable group, e.g., "
                                        "\"'state/foo' 'state/bar'; 'state/baz'\".";

  options.set<std::vector<std::vector<VariableName>>>("residual_groups");
  options.set("residual_groups").doc() = "Optional grouping of residual variables. "
                                         "Each inner list defines one variable group, e.g., "
                                         "\"'state/foo' 'state/bar'; 'state/baz'\".";

  return options;
}

ModelNonlinearSystem::ModelNonlinearSystem(const OptionSet & options)
  : EquationSystem(options),
    ParameterStore(this),
    BufferStore(this),
    _unknown_groups(
        options.user_specified("unknown_groups")
            ? std::optional(options.get<std::vector<std::vector<VariableName>>>("unknown_groups"))
            : std::nullopt),
    _residual_groups(
        options.user_specified("residual_groups")
            ? std::optional(options.get<std::vector<std::vector<VariableName>>>("residual_groups"))
            : std::nullopt),
    _model(get_model("model"))
{
  if (_unknown_groups.has_value() != _residual_groups.has_value())
    throw NEMLException(
        "Options 'unknown_groups' and 'residual_groups' must be specified together.");
}

void
ModelNonlinearSystem::setup()
{
  EquationSystem::setup();
  NonlinearSystem::init();

  if (host() == this)
  {
    _model->link_output_variables();
    _model->link_input_variables();
  }

  // unknown variables should be marked mutable, as they will be updated during the nonlinear solve
  for (const auto & vname : full_umap())
    _model->input_variable(vname).set_mutable(true);
}

void
ModelNonlinearSystem::to(const TensorOptions & options)
{
  _model->to(options);
  send_parameters_to(options);
  send_buffers_to(options);
}

std::vector<std::vector<LabeledAxisAccessor>>
ModelNonlinearSystem::setup_umap()
{
  return grouped_unknowns();
}

std::vector<std::vector<TensorShape>>
ModelNonlinearSystem::setup_intmd_ulayout()
{
  std::vector<std::vector<TensorShape>> intmd_ulayout;
  for (const auto & group : grouped_unknowns())
  {
    std::vector<TensorShape> group_intmd_ulayout;
    group_intmd_ulayout.reserve(group.size());
    for (const auto & vname : group)
      group_intmd_ulayout.emplace_back(model().input_variable(vname).intmd_sizes());
    intmd_ulayout.push_back(std::move(group_intmd_ulayout));
  }
  return intmd_ulayout;
}

std::vector<std::vector<TensorShape>>
ModelNonlinearSystem::setup_ulayout()
{
  std::vector<std::vector<TensorShape>> ulayout;
  for (const auto & group : grouped_unknowns())
  {
    std::vector<TensorShape> group_ulayout;
    group_ulayout.reserve(group.size());
    for (const auto & vname : group)
      group_ulayout.emplace_back(model().input_variable(vname).base_sizes());
    ulayout.push_back(std::move(group_ulayout));
  }
  return ulayout;
}

std::vector<std::vector<LabeledAxisAccessor>>
ModelNonlinearSystem::setup_bmap()
{
  return grouped_residuals();
}

std::vector<std::vector<TensorShape>>
ModelNonlinearSystem::setup_intmd_blayout()
{
  std::vector<std::vector<TensorShape>> intmd_blayout;
  for (const auto & group : grouped_residuals())
  {
    std::vector<TensorShape> group_intmd_blayout;
    group_intmd_blayout.reserve(group.size());
    for (const auto & vname : group)
      group_intmd_blayout.emplace_back(model().output_variable(vname).intmd_sizes());
    intmd_blayout.push_back(std::move(group_intmd_blayout));
  }
  return intmd_blayout;
}

std::vector<std::vector<TensorShape>>
ModelNonlinearSystem::setup_blayout()
{
  std::vector<std::vector<TensorShape>> blayout;
  for (const auto & group : grouped_residuals())
  {
    std::vector<TensorShape> group_blayout;
    group_blayout.reserve(group.size());
    for (const auto & vname : group)
      group_blayout.emplace_back(model().output_variable(vname).base_sizes());
    blayout.push_back(std::move(group_blayout));
  }
  return blayout;
}

std::vector<LabeledAxisAccessor>
ModelNonlinearSystem::setup_gmap()
{
  std::vector<LabeledAxisAccessor> gmap;
  for (const auto & [vname, var] : _model->input_variables())
    if (!var->is_state())
      gmap.push_back(vname);
  return gmap;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_intmd_glayout()
{
  std::vector<TensorShape> intmd_glayout;
  for (const auto & [vname, var] : _model->input_variables())
    if (!var->is_state())
      intmd_glayout.emplace_back(var->intmd_sizes());
  return intmd_glayout;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_glayout()
{
  std::vector<TensorShape> glayout;
  for (const auto & [vname, var] : _model->input_variables())
    if (!var->is_state())
      glayout.emplace_back(var->base_sizes());
  return glayout;
}

void
ModelNonlinearSystem::set_u(const SparseTensorList & u)
{
  _model->assign_input(full_umap(), u);
  u_changed();
}

void
ModelNonlinearSystem::set_g(const SparseTensorList & g)
{
  _model->assign_input(gmap(), g);
  g_changed();
}

SparseTensorList
ModelNonlinearSystem::u() const
{
  return _model->collect_input(full_umap());
}

SparseTensorList
ModelNonlinearSystem::g() const
{
  return _model->collect_input(gmap());
}

void
ModelNonlinearSystem::assemble(SparseTensorList * A, SparseTensorList * B, SparseTensorList * b)
{
  {
    AssemblyingNonlinearSystem assembling_nl_sys(!B);
    _model->forward_maybe_jit(
        b && !_b_up_to_date, (A && !_A_up_to_date) || (B && !_B_up_to_date), false);
  }

  if (b)
    // remember b := -r
    *b = -_model->collect_output(full_bmap());

  if (A)
    *A = _model->collect_output_derivatives(full_bmap(), full_umap());

  if (B)
    *B = _model->collect_output_derivatives(full_bmap(), gmap());
}

void
ModelNonlinearSystem::pre_assemble(bool A, bool B, bool b)
{
  NonlinearSystem::pre_assemble(A, B, b);

  if (host() == this)
    _model->zero_undefined_input();
}

void
ModelNonlinearSystem::post_assemble(bool A, bool B, bool b)
{
  NonlinearSystem::post_assemble(A, B, b);

  if (host() == this)
  {
    u_changed();
    g_changed();
    _model->clear_input();
    _model->clear_output();
  }
}

std::vector<std::vector<LabeledAxisAccessor>>
ModelNonlinearSystem::grouped_unknowns() const
{
  return build_grouped_map(
      _model->input_variables(),
      _unknown_groups,
      [](const auto & var) { return var.is_state(); },
      "unknown_groups",
      "state");
}

std::vector<std::vector<LabeledAxisAccessor>>
ModelNonlinearSystem::grouped_residuals() const
{
  return build_grouped_map(
      _model->output_variables(),
      _residual_groups,
      [](const auto & var) { return var.is_residual(); },
      "residual_groups",
      "residual");
}

} // namespace neml2
