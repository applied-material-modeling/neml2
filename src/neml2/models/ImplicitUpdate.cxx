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

#include "neml2/models/ImplicitUpdate.h"
#include "neml2/misc/assertions.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/models/ModelNonlinearSystem.h"

namespace neml2
{
register_NEML2_object(ImplicitUpdate);

OptionSet
ImplicitUpdate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Update an implicit model by solving the underlying nonlinear system of equations.";

  options.set<std::string>("equation_system");
  options.set("equation_system").doc() = "The nonlinear system of equations to solve";

  options.set<std::string>("solver");
  options.set("solver").doc() = "Solver used to solve the nonlinear system of equations";

  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _sys(get_es<ModelNonlinearSystem>("equation_system")),
    _solver(get_solver<NonlinearSolver>("solver"))
{
  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  const auto & model = _sys->model();
  for (auto && [name, var] : model.input_variables())
    clone_input_variable(*var);
  for (auto && [name, var] : model.output_variables())
    clone_output_variable(*var, name.remount(STATE));
}

void
ImplicitUpdate::to(const TensorOptions & options)
{
  Model::to(options);
  _sys->to(options);
  _solver->to(options);
}

void
ImplicitUpdate::link_input_variables()
{
  Model::link_input_variables();
  for (auto && [name, var] : _sys->model().input_variables())
    var->ref(input_variable(name));
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Input variable values may have been changed outside the forward operator, so let's notify the
  // system about the potential changes, just to stay on the safe side.
  _sys->u_changed();
  _sys->g_changed();

  // Solve for the next state
  const auto res = _solver->solve(*_sys);
  _last_iterations = res.iterations;
  neml_assert(res.ret == NonlinearSolver::RetCode::SUCCESS, "Nonlinear solve failed.");

  if (out)
    assign_output(_sys->umap(), _sys->u());

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    const auto du_dg = _solver->linear_solver->ift(*_sys);

    // assign derivatives back
    assign_output_derivatives(_sys->umap(), _sys->gmap(), du_dg);
  }
}
} // namespace neml2
