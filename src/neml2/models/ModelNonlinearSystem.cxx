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

  return options;
}

ModelNonlinearSystem::ModelNonlinearSystem(const OptionSet & options)
  : NonlinearSystem(options),
    ParameterStore(this),
    BufferStore(this),
    _model(get_model("model"))
{
}

void
ModelNonlinearSystem::setup()
{
  NonlinearSystem::setup();

  if (host() == this)
  {
    _model->link_output_variables();
    _model->link_input_variables();
  }

  // unknown variables should be marked mutable, as they will be updated during the nonlinear solve
  for (const auto & vname : umap())
    _model->input_variable(vname).set_mutable(true);
}

void
ModelNonlinearSystem::to(const TensorOptions & options)
{
  _model->to(options);
  send_parameters_to(options);
  send_buffers_to(options);
}

std::vector<LabeledAxisAccessor>
ModelNonlinearSystem::setup_umap()
{
  std::vector<LabeledAxisAccessor> umap;
  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_state())
      umap.push_back(vname);
  return umap;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_intmd_ulayout()
{
  std::vector<TensorShape> intmd_ulayout;
  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_state())
      intmd_ulayout.emplace_back(var->intmd_sizes());
  return intmd_ulayout;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_ulayout()
{
  std::vector<TensorShape> ulayout;
  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_state())
      ulayout.emplace_back(var->base_sizes());
  return ulayout;
}

std::vector<LabeledAxisAccessor>
ModelNonlinearSystem::setup_bmap()
{
  std::vector<LabeledAxisAccessor> bmap;
  for (const auto & [vname, var] : _model->output_variables())
    if (var->is_residual())
      bmap.push_back(vname);
  return bmap;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_intmd_blayout()
{
  std::vector<TensorShape> intmd_blayout;
  for (const auto & [vname, var] : _model->output_variables())
    if (var->is_residual())
      intmd_blayout.emplace_back(var->intmd_sizes());
  return intmd_blayout;
}

std::vector<TensorShape>
ModelNonlinearSystem::setup_blayout()
{
  std::vector<TensorShape> blayout;
  for (const auto & [vname, var] : _model->output_variables())
    if (var->is_residual())
      blayout.emplace_back(var->base_sizes());
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
  _model->assign_input(umap(), u);
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
  return _model->collect_input(umap());
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
    _model->forward_maybe_jit(true, (A && !_A_up_to_date) || (B && !_B_up_to_date), false);
  }

  if (b)
    // remember b := -r
    *b = -_model->collect_output(bmap());

  if (A)
    *A = _model->collect_output_derivatives(bmap(), umap());

  if (B)
    *B = _model->collect_output_derivatives(bmap(), gmap());
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
