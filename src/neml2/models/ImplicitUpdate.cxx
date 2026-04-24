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

  options.add<std::string>("equation_system", "The nonlinear system of equations to solve");
  options.add<std::string>("solver", "Solver used to solve the nonlinear system of equations");

  // No jitting :/
  options.set<bool>("jit", false);
  options.suppress("jit");

  // deprecated
  options.add_optional<std::string>("implicit_model",
                                    "Deprecated option. Use 'equation_system' instead.");

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _sys(get_es<ModelNonlinearSystem>("equation_system")),
    _solver(get_solver<NonlinearSolver>("solver"))
{
  neml_assert(!options.user_specified("implicit_model"),
              "The 'implicit_model' option is deprecated. Use 'equation_system' instead. Refer to "
              "https://applied-material-modeling.github.io/neml2/migration-200-210.html#eqsys for "
              "more information.");

  auto ulayout = _sys->ulayout();
  auto blayout = _sys->blayout();
  neml_assert(
      blayout.ngroup() == ulayout.ngroup(),
      "The unknown variables and residuals of the nonlinear system must have the same number of "
      "groups. Got ",
      ulayout.ngroup(),
      " and ",
      blayout.ngroup(),
      ".");
  neml_assert(blayout.nvar() == ulayout.nvar(),
              "The number of unknown variables and residuals of the nonlinear system must be the "
              "same. Got ",
              ulayout.nvar(),
              " and ",
              blayout.nvar(),
              ".");

  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  const auto model = _sys->model_ptr();
  register_model(model, /*merge_input=*/true);
  for (std::size_t i = 0; i < blayout.nvar(); ++i)
  {
    const auto & u = model->input_variable(ulayout.var(i));
    clone_output_variable(u);
  }
}

void
ImplicitUpdate::to(const TensorOptions & options)
{
  Model::to(options);
  _sys->to(options);
  _solver->to(options);
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
