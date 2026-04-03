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
#include "neml2/base/VariableName.h"
#include "neml2/base/MultiEnumSelection.h"
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/equation_systems/EquationSystem.h"
#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"
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

  options.add<std::string>("model", "The Model defining this nonlinear system.");

  options.add<std::vector<std::vector<VariableName>>>(
      "unknowns", "Ordering and grouping of unknowns. Each inner list defines one variable group.");
  options.add<std::vector<std::vector<VariableName>>>(
      "residuals",
      "Ordering and grouping of residual variables. Each inner list defines one variable group.");

  MultiEnumSelection istr_selection({"DENSE", "BLOCK"},
                                    {static_cast<int>(AxisLayout::IStructure::DENSE),
                                     static_cast<int>(AxisLayout::IStructure::BLOCK)},
                                    {"DENSE"});
  options.add<MultiEnumSelection>(
      "istructure",
      istr_selection,
      "Optional IStructure for each variable group. If not provided, defaults to DENSE. If only "
      "one IStructure is provided, it will be applied to all groups.");

  return options;
}

ModelNonlinearSystem::ModelNonlinearSystem(const OptionSet & options)
  : EquationSystem(options),
    ParameterStore(this),
    BufferStore(this),
    _unknown_groups(options.get<std::vector<std::vector<VariableName>>>("unknowns")),
    _residual_groups(options.get<std::vector<std::vector<VariableName>>>("residuals")),
    _unknown_istrs(options.get<MultiEnumSelection>("istructure").as<AxisLayout::IStructure>()),
    _residual_istrs(options.get<MultiEnumSelection>("istructure").as<AxisLayout::IStructure>()),
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
  const auto ul = ulayout();
  for (std::size_t i = 0; i < ul.nvar(); i++)
    _model->input_variable(ul.var(i)).set_mutable(true);
}

void
ModelNonlinearSystem::to(const TensorOptions & options)
{
  _model->to(options);
  send_parameters_to(options);
  send_buffers_to(options);
}

std::shared_ptr<AxisLayout>
ModelNonlinearSystem::setup_ulayout()
{
  // gather intmd/base shapes for each variable in the layout
  std::vector<TensorShape> intmd_shapes, base_shapes;
  for (const auto & vars : _unknown_groups)
    for (const auto & vname : vars)
    {
      const auto & var = model().input_variable(vname);
      intmd_shapes.emplace_back(var.intmd_sizes());
      base_shapes.emplace_back(var.base_sizes());
    }

  // IStructure
  std::vector<AxisLayout::IStructure> istrs = _unknown_istrs;
  if (istrs.size() == 1 && _unknown_groups.size() > 1)
    istrs.resize(_unknown_groups.size(), istrs[0]);

  return std::make_shared<AxisLayout>(_unknown_groups, intmd_shapes, base_shapes, istrs);
}

std::shared_ptr<AxisLayout>
ModelNonlinearSystem::setup_glayout()
{
  std::vector<std::vector<VariableName>> vars(1);
  std::vector<TensorShape> intmd_shapes, base_shapes;
  for (const auto & [vname, var] : _model->input_variables())
  {
    // skip if this variable is already in ulayout
    bool in_ulayout = false;
    for (const auto & uvars : _unknown_groups)
      if (std::find(uvars.begin(), uvars.end(), vname) != uvars.end())
      {
        in_ulayout = true;
        break;
      }
    if (in_ulayout)
      continue;

    vars[0].push_back(vname);
    intmd_shapes.emplace_back(var->intmd_sizes());
    base_shapes.emplace_back(var->base_sizes());
  }

  // TODO: take IStructure from input file options
  std::vector<AxisLayout::IStructure> istrs(1, AxisLayout::IStructure::DENSE);

  return std::make_shared<AxisLayout>(vars, intmd_shapes, base_shapes, istrs);
}

std::shared_ptr<AxisLayout>
ModelNonlinearSystem::setup_blayout()
{
  // gather intmd/base shapes for each variable in the layout
  std::vector<TensorShape> intmd_shapes, base_shapes;
  for (const auto & vars : _residual_groups)
    for (const auto & vname : vars)
    {
      const auto & var = model().output_variable(vname);
      intmd_shapes.emplace_back(var.intmd_sizes());
      base_shapes.emplace_back(var.base_sizes());
    }

  // IStructure
  std::vector<AxisLayout::IStructure> istrs = _residual_istrs;
  if (istrs.size() == 1 && _residual_groups.size() > 1)
    istrs.resize(_residual_groups.size(), istrs[0]);

  return std::make_shared<AxisLayout>(_residual_groups, intmd_shapes, base_shapes, istrs);
}

void
ModelNonlinearSystem::update_layouts()
{
  if (_layouts_updated)
    return;

  // After the first evaluation, we have the correct intmd shapes for the variables, so we can
  // update layouts accordingly. We need to do this before assembling which depends on the accurate
  // layouts.
  const auto & m = this->model();
  auto & ul = *_ulayout;
  auto & gl = *_glayout;
  auto & bl = *_blayout;
  std::vector<TensorShape> uis(ul.nvar()), gis(gl.nvar()), bis(bl.nvar());
  for (std::size_t i = 0; i < ul.nvar(); i++)
    uis[i] = m.input_variable(ul.var(i)).intmd_sizes();
  for (std::size_t i = 0; i < gl.nvar(); i++)
    gis[i] = m.input_variable(gl.var(i)).intmd_sizes();
  for (std::size_t i = 0; i < bl.nvar(); i++)
    bis[i] = m.output_variable(bl.var(i)).intmd_sizes();
  ul.update_intmd_shapes(uis);
  gl.update_intmd_shapes(gis);
  bl.update_intmd_shapes(bis);
  _layouts_updated = true;
}

void
ModelNonlinearSystem::set_u(const AssembledVector & u)
{
  _model->assign_input(u.disassemble());
  u_changed();
}

void
ModelNonlinearSystem::set_g(const AssembledVector & g)
{
  _model->assign_input(g.disassemble(), /*allow_nonexistent=*/true);
  g_changed();
}

AssembledVector
ModelNonlinearSystem::u() const
{
  return _model->collect_input(ulayout()).assemble();
}

AssembledVector
ModelNonlinearSystem::g() const
{
  return _model->collect_input(glayout()).assemble();
}

void
ModelNonlinearSystem::assemble(AssembledMatrix * A, AssembledMatrix * B, AssembledVector * b)
{
  {
    AssemblyingNonlinearSystem assembling_nl_sys(!B);
    _model->forward_maybe_jit(
        b && !_b_up_to_date, (A && !_A_up_to_date) || (B && !_B_up_to_date), false);
  }

  update_layouts();

  if (b)
    // remember b := -r
    *b = -_model->collect_output(blayout()).assemble();

  if (A)
    *A = _model->collect_output_derivatives(blayout(), ulayout()).assemble();

  if (B)
    *B = _model->collect_output_derivatives(blayout(), glayout()).assemble();
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
