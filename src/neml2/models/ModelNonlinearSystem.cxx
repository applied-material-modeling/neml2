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
#include "neml2/models/Model.h"
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"

namespace neml2
{

ModelNonlinearSystem::ModelNonlinearSystem(Model * model, bool assembly_guard)
  : _model(model),
    _assembly_guard(assembly_guard)
{
  // Note: These data structures we are setting here serve the same purpose as LabeledAxis, and yes,
  // we are storing redundant information. This is because we are transitioning away from
  // LabeledAxis. In future versions, we will remove LabeledAxis and only use these data structures.
  //
  // Also note: only nonlinear systems need to define these maps. Regular feed-forward models do not
  // need to define these maps. This is an important distinction from LabeledAxis which is always
  // defined.
  //
  // Another note: Right now we can "smartly" determine these maps based on variable subaxes. In the
  // future, since we are removing LabeledAxis, we will need let the user explicitly define these
  // maps, i.e., from within the input files.

  for (const auto & [vname, var] : _model->input_variables())
    if (var->is_state())
    {
      _umap.push_back(vname);
      _ulayout.emplace_back(var->base_sizes());
    }
    else if (var->is_force())
    {
      _gmap.push_back(vname);
      _glayout.emplace_back(var->base_sizes());
    }
    else if (var->is_old_state())
    {
      _unmap.push_back(vname);
      _unlayout.emplace_back(var->base_sizes());
    }
    else if (var->is_old_force())
    {
      _gnmap.push_back(vname);
      _gnlayout.emplace_back(var->base_sizes());
    }

  for (const auto & [vname, var] : _model->output_variables())
    if (var->is_residual())
    {
      _bmap.push_back(vname);
      _blayout.emplace_back(var->base_sizes());
    }

  // setup old-to-current maps
  auto setup_old_to_current = [](const std::vector<LabeledAxisAccessor> & old_map,
                                 const std::vector<LabeledAxisAccessor> & current_map,
                                 std::vector<int64_t> & old_to_current)
  {
    old_to_current.resize(old_map.size(), -1);
    for (std::size_t i = 0; i < old_map.size(); ++i)
    {
      const auto & uname = old_map[i].current();
      auto itr = std::find(current_map.begin(), current_map.end(), uname);
      if (itr != current_map.end())
        old_to_current[i] = std::distance(current_map.begin(), itr);
    }
  };
  setup_old_to_current(_unmap, _umap, _un_to_u);
  setup_old_to_current(_gnmap, _gmap, _gn_to_g);

  // setup current-to-old maps
  auto setup_current_to_old = [](const std::vector<LabeledAxisAccessor> & current_map,
                                 const std::vector<LabeledAxisAccessor> & old_map,
                                 std::vector<int64_t> & current_to_old)
  {
    current_to_old.resize(current_map.size(), -1);
    for (std::size_t i = 0; i < current_map.size(); ++i)
    {
      const auto & unname = current_map[i].old();
      auto itr = std::find(old_map.begin(), old_map.end(), unname);
      if (itr != old_map.end())
        current_to_old[i] = std::distance(old_map.begin(), itr);
    }
  };
  setup_current_to_old(_umap, _unmap, _u_to_un);
  setup_current_to_old(_gmap, _gnmap, _g_to_gn);

  // umap variables should be marked mutable, as they will be updated during the nonlinear solve
  for (const auto & vname : _umap)
    _model->input_variable(vname).set_mutable(true);
}

void
ModelNonlinearSystem::set_u(const HVector & u)
{
  _model->assign_input(umap(), u);
  _A_up_to_date = false;
  _b_up_to_date = false;
}

void
ModelNonlinearSystem::set_un(const HVector & un)
{
  _model->assign_input(unmap(), un);
  _A_up_to_date = false;
  _b_up_to_date = false;
}

void
ModelNonlinearSystem::set_g(const HVector & g)
{
  _model->assign_input(gmap(), g);
  _A_up_to_date = false;
  _b_up_to_date = false;
}

void
ModelNonlinearSystem::set_gn(const HVector & gn)
{
  _model->assign_input(gnmap(), gn);
  _A_up_to_date = false;
  _b_up_to_date = false;
}

HVector
ModelNonlinearSystem::u() const
{
  return HVector(_model->collect_input(umap()));
}

HVector
ModelNonlinearSystem::un() const
{
  return HVector(_model->collect_input(unmap()));
}

HVector
ModelNonlinearSystem::g() const
{
  return HVector(_model->collect_input(gmap()));
}

HVector
ModelNonlinearSystem::gn() const
{
  return HVector(_model->collect_input(gnmap()));
}

void
ModelNonlinearSystem::assemble(HMatrix * A, HVector * b)
{
  {
    AssemblyingNonlinearSystem assembling_nl_sys(_model, _assembly_guard);
    _model->forward_maybe_jit(b && !_b_up_to_date, A && !_A_up_to_date, false);
    _A_up_to_date |= bool(A);
    _b_up_to_date |= bool(b);
  }

  if (b)
    // remember b := -r(u)
    *b = -_model->collect_output(bmap());

  if (A)
    *A = _model->collect_output_derivatives(bmap(), umap());
}

} // namespace neml2
