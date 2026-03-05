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
#include "neml2/base/LabeledAxisAccessor.h"
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/equation_systems/EquationSystem.h"
#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/misc/types.h"
#include "neml2/models/Model.h"

namespace neml2
{
register_NEML2_object_alias(ModelNonlinearSystem, "NonlinearSystem");

OptionSet
ModelNonlinearSystem::expected_options()
{
  OptionSet options = EquationSystem::expected_options();

  options.doc() = "A nonlinear system defined by a Model.";

  options.set<std::string>("model") = "model";
  options.set("model").doc() = "The Model defining this nonlinear system.";

  options.set<std::vector<std::vector<VariableName>>>("unknowns");
  options.set("unknowns").doc() =
      "Optional ordering and grouping of unknowns. Each inner list defines one variable group.";

  options.set<std::vector<std::vector<VariableName>>>("residuals");
  options.set("residuals").doc() = "Optional ordering and grouping of residual variables. Each "
                                   "inner list defines one variable group.";

  return options;
}

ModelNonlinearSystem::ModelNonlinearSystem(const OptionSet & options)
  : EquationSystem(options),
    ParameterStore(this),
    BufferStore(this),
    _unknown_groups(options.get<std::vector<std::vector<VariableName>>>("unknowns")),
    _residual_groups(options.get<std::vector<std::vector<VariableName>>>("residuals")),
    _model(get_model("model"))
{
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
  const auto & ul = *ulayout();
  for (std::size_t i = 0; i < ul.size(); ++i)
    for (const auto & vname : ul.vars[i])
      _model->input_variable(vname).set_mutable(true);
}

void
ModelNonlinearSystem::to(const TensorOptions & options)
{
  _model->to(options);
  send_parameters_to(options);
  send_buffers_to(options);
}

std::vector<std::shared_ptr<AxisLayout>>
ModelNonlinearSystem::setup_ulayout()
{
  // one single group if unknowns not specified
  if (_unknown_groups.empty())
  {
    _unknown_groups.resize(1);
    for (const auto & [vname, var] : _model->input_variables())
      if (vname.is_state())
        _unknown_groups[0].push_back(vname);
  }

  // create layout for each variable group
  std::vector<std::shared_ptr<AxisLayout>> layout;
  for (std::size_t i = 0; i < _unknown_groups.size(); ++i)
  {
    const auto & vars = _unknown_groups[i];
    std::vector<TensorShape> intmd_shapes, base_shapes;
    intmd_shapes.reserve(vars.size());
    base_shapes.reserve(vars.size());
    for (const auto & vname : vars)
    {
      neml_assert(vname.is_state(), vname, " is not a state variable.");
      const auto & var = model().input_variable(vname);
      intmd_shapes.emplace_back(var.intmd_sizes());
      base_shapes.emplace_back(var.base_sizes());
    }
    layout.emplace_back(std::make_shared<AxisLayout>(AxisLayout{vars, intmd_shapes, base_shapes}));
  }
  return layout;
}

std::shared_ptr<AxisLayout>
ModelNonlinearSystem::setup_glayout()
{
  std::vector<VariableName> vars;
  std::vector<TensorShape> intmd_shapes, base_shapes;
  for (const auto & [vname, var] : _model->input_variables())
    if (!vname.is_state())
    {
      vars.push_back(vname);
      intmd_shapes.emplace_back(var->intmd_sizes());
      base_shapes.emplace_back(var->base_sizes());
    }

  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes);
}

std::vector<std::shared_ptr<AxisLayout>>
ModelNonlinearSystem::setup_blayout()
{
  // one single group if residuals not specified
  if (_residual_groups.empty())
  {
    _residual_groups.resize(1);
    for (const auto & [vname, var] : _model->input_variables())
      if (vname.is_state())
        _residual_groups[0].push_back(vname);
  }

  // create layout for each variable group
  std::vector<std::shared_ptr<AxisLayout>> layout;
  for (std::size_t i = 0; i < _residual_groups.size(); ++i)
  {
    const auto & vars = _residual_groups[i];
    std::vector<TensorShape> intmd_shapes, base_shapes;
    intmd_shapes.reserve(vars.size());
    base_shapes.reserve(vars.size());
    for (const auto & vname : vars)
    {
      neml_assert(vname.is_residual(), vname, " is not a residual variable.");
      const auto & var = model().output_variable(vname);
      intmd_shapes.emplace_back(var.intmd_sizes());
      base_shapes.emplace_back(var.base_sizes());
    }
    layout.emplace_back(std::make_shared<AxisLayout>(AxisLayout{vars, intmd_shapes, base_shapes}));
  }
  return layout;
}

void
ModelNonlinearSystem::set_u(const SparseTensorList & u, std::size_t group_idx)
{
  _model->assign_input(_ulayout[group_idx]->vars, u);
  u_changed();
}

void
ModelNonlinearSystem::set_g(const SparseTensorList & g)
{
  _model->assign_input(_glayout->vars, g);
  g_changed();
}

SparseVector
ModelNonlinearSystem::u(std::size_t group_idx) const
{
  return {_model->collect_input(_ulayout[group_idx]->vars), _ulayout[group_idx]};
}

SparseVector
ModelNonlinearSystem::g() const
{
  return {_model->collect_input(_glayout->vars), _glayout};
}

void
ModelNonlinearSystem::assemble(SparseTensorList * A,
                               SparseTensorList * B,
                               SparseTensorList * b,
                               std::size_t bgroup_idx,
                               std::size_t ugroup_idx)
{
  {
    AssemblyingNonlinearSystem assembling_nl_sys(!B);
    _model->forward_maybe_jit(
        b && !_b_up_to_date, (A && !_A_up_to_date) || (B && !_B_up_to_date), false);
  }

  if (b)
    // remember b := -r
    *b = -_model->collect_output(_blayout[bgroup_idx]->vars);

  if (A)
    *A = _model->collect_output_derivatives(_blayout[bgroup_idx]->vars, _ulayout[ugroup_idx]->vars);

  if (B)
    *B = _model->collect_output_derivatives(_blayout[bgroup_idx]->vars, _glayout->vars);
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

} // namespace neml2
