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
#include "neml2/models/Assembler.h"
#include "neml2/solvers/NonlinearSolver.h"
#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/functions/linalg/lu_solve.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(ImplicitUpdate);

OptionSet
ImplicitUpdate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Update an implicit model by solving the underlying implicit system of equations.";

  options.set<std::string>("implicit_model");
  options.set("implicit_model").doc() =
      "The model defining the (nonlinear) system of equations to be solved";

  options.set<std::string>("solver");
  options.set("solver").doc() = "Solver used to solve the (nonlinear) system of equations";

  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _model(register_model("implicit_model", /*nonlinear=*/true)),
    _solver(get_solver<NonlinearSolver>("solver"))
{
  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model. This has
  //      already been taken care of by the `register_model` call.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  for (auto && [name, var] : _model.output_variables())
    clone_output_variable(*var, name.remount(STATE));
}

void
ImplicitUpdate::diagnose() const
{
  Model::diagnose();
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // The trial state is used as the initial guess
  const auto x0 = _model.get_solution();

  // Solve for the next state
  NonlinearSolver::Result res = _solver->solve(_model, x0);
  _last_iterations = res.iterations;
  neml_assert(res.ret == NonlinearSolver::RetCode::SUCCESS, "Nonlinear solve failed.");

  if (out)
    assign_output(_model.unknown_ordering(), res.solution);

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    // IFT requires the Jacobian evaluated at the solution:
    _model.forward_maybe_jit(false, true, false);
    const auto dr_ds =
        _model.collect_output_derivatives(_model.residual_ordering(), _model.unknown_ordering());
    const auto dr_dp =
        _model.collect_output_derivatives(_model.residual_ordering(), _model.prescribed_ordering());

    // ds/dp = - (dr/ds)^{-1} (dr/dp)
    const auto ds_dp = -_solver->linear_solver->solve(dr_ds, dr_dp);
    assign_output_derivatives(_model.unknown_ordering(), _model.prescribed_ordering(), ds_dp);
  }
}
} // namespace neml2
