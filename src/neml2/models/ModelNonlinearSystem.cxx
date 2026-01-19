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

namespace neml2
{

ModelNonlinearSystem::ModelNonlinearSystem(Model * model)
  : _model(model)
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
      _rmap.push_back(vname);
      _rlayout.emplace_back(var->base_sizes());
    }
}

void
ModelNonlinearSystem::set_x(const HVector & x)
{
  _model->assign_input(umap(), x);
}

HVector
ModelNonlinearSystem::x() const
{
  return HVector(_model->collect_input(umap()));
}

void
ModelNonlinearSystem::assemble(HMatrix * A, HVector * b)
{
  _model->currently_assembling_nonlinear_system(true);
  _model->forward_maybe_jit(b, A, false);
  _model->currently_assembling_nonlinear_system(false);

  if (b)
    // remember that b := -r(x)
    *b = -_model->collect_output(rmap());

  if (A)
    *A = _model->collect_output_derivatives(rmap(), umap());
}

} // namespace neml2
