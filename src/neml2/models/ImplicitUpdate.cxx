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
#include "neml2/models/ModelNonlinearSystem.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"

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

  options.set<std::string>("initial_guess_model") = "";
  options.set("initial_guess_model").doc() =
      "Optional model to compute a customized initial guess for the Newton solve. When provided, "
      "this model is evaluated once before the Newton loop begins. Its output variables that "
      "correspond to Newton unknowns (the umap) are extracted and used as the Newton starting "
      "point, replacing the default predictor (old_state values). The model's inputs must be a "
      "subset of the equation system's given variables (gmap), so they are automatically "
      "available when the solver is called.";

  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;

  // deprecated
  options.set<std::string>("implicit_model");
  options.set("implicit_model").doc() = "Deprecated option. Use 'equation_system' instead.";

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _sys(get_es<ModelNonlinearSystem>("equation_system")),
    _solver(get_solver<NonlinearSolver>("solver")),
    _initial_guess_model(nullptr)
{
  neml_assert(!options.user_specified("implicit_model"),
              "The 'implicit_model' option is deprecated. Use 'equation_system' instead. Refer to "
              "https://applied-material-modeling.github.io/neml2/migration-200-210.html#eqsys for "
              "more information.");

  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  const auto model = _sys->model_ptr();
  register_model(model, /*merge_input=*/true);
  for (auto && [name, var] : model->output_variables())
    clone_output_variable(*var, name.remount(STATE));

  // Register the optional initial guess model.
  // Do NOT merge its inputs (merge_input=false): the variables it needs (forces/E,
  // old_state/Ep, old_state/ep, etc.) are already registered by the equation system model.
  // We still register it so setup() is called on it. Its inputs are assigned manually in
  // set_value() from ImplicitUpdate's own input variable map.
  const auto & ig_name = options.get<std::string>("initial_guess_model");
  if (!ig_name.empty())
  {
    _initial_guess_model = get_model<Model>(ig_name);
    register_model(_initial_guess_model, /*merge_input=*/false);
  }
}

void
ImplicitUpdate::to(const TensorOptions & options)
{
  Model::to(options);
  _sys->to(options);
  _solver->to(options);
  if (_initial_guess_model)
    _initial_guess_model->to(options);
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // Input variable values may have been changed outside the forward operator, so let's notify the
  // system about the potential changes, just to stay on the safe side.
  _sys->u_changed();
  _sys->g_changed();

  // If an initial guess model is provided, evaluate it to get a better Newton starting point.
  // Manually assign the inputs it needs from ImplicitUpdate's own input variable map
  // (since we registered with merge_input=false to avoid duplicate variable errors).
  if (_initial_guess_model)
  {
    ValueMap ig_inputs;
    for (const auto & [vname, var] : _initial_guess_model->input_variables())
      if (input_variables().count(vname))
        ig_inputs[vname] = input_variable(vname).tensor();
    _initial_guess_model->assign_input(ig_inputs);
    _initial_guess_model->forward_maybe_jit(/*out=*/true, /*dout=*/false, /*d2out=*/false);
    const auto ig_u = _initial_guess_model->collect_output(_sys->umap());
    _sys->set_u(ig_u);
  }

  // Solve for the next state
  const auto res = _solver->solve(*_sys);
  _last_iterations = res.iterations;
  neml_assert(res.ret == NonlinearSolver::RetCode::SUCCESS, "Nonlinear solve failed.");

  if (out)
    assign_output(_sys->u().disassemble());

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    auto [A, B] = _sys->A_and_B();
    auto du_dg = -_solver->linear_solver->solve(A, B);

    // assign derivatives back
    assign_output_derivatives(du_dg.disassemble());
  }
}
} // namespace neml2
