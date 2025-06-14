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
#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/functions/linalg/lu_solve.h"
#include "neml2/base/guards.h"
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
      "The implicit model defining the implicit system of equations to be solved";

  options.set<std::string>("solver");
  options.set("solver").doc() = "Solver used to solve the implicit system";

  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _model(register_model(options.get<std::string>("implicit_model"),
                          /*nonlinear=*/true)),
    _solver(factory()->get_object<NonlinearSolver>("Solvers", options.get<std::string>("solver")))
{
  neml_assert(_model.output_axis().has_residual(),
              "The implicit model'",
              _model.name(),
              "' registered in '",
              name(),
              "' does not have the residual output axis.");
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
  diagnostic_assert(_model.output_axis().nsubaxis() == 1,
                    "The implicit model's output contains non-residual subaxis:\n",
                    _model.output_axis());
  diagnostic_assert(_model.input_axis().has_state(),
                    "The implicit model's input does not have a state subaxis:\n",
                    _model.input_axis());
  diagnostic_assert(!_model.input_axis().has_residual(),
                    "The implicit model's input cannot have a residual subaxis:\n",
                    _model.input_axis());
  diagnostic_assert(
      _model.input_axis().subaxis(STATE) == _model.output_axis().subaxis(RESIDUAL),
      "The implicit model should have conformal trial state and residual. The input state "
      "subaxis is\n",
      _model.input_axis().subaxis(STATE),
      "\nThe output residual subaxis is\n",
      _model.output_axis().subaxis(RESIDUAL));
}

void
ImplicitUpdate::link_output_variables()
{
  Model::link_output_variables();
  for (auto && [name, var] : output_variables())
    var->ref(input_variable(name), /*ref_is_mutable=*/true);
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // The trial state is used as the initial guess
  const auto sol_assember = VectorAssembler(_model.input_axis().subaxis(STATE));
  auto x0 = NonlinearSystem::Sol<false>(sol_assember.assemble_by_variable(_model.collect_input()));

  // Perform automatic scaling (using the trial state)
  // TODO: Add an interface to allow user to specify where (and when) to evaluate the Jacobian for
  // automatic scaling.
  _model.init_scaling(x0, _solver->verbose);

  // Solve for the next state
  NonlinearSolver::Result res;
  {
    SolvingNonlinearSystem solving;
    res = _solver->solve(_model, x0);
    neml_assert(res.ret == NonlinearSolver::RetCode::SUCCESS, "Nonlinear solve failed.");
  }

  if (out)
  {
    // You may be tempted to assign the solution, i.e., res.solution, to the output variables. But
    // we don't have to. Think about it: The output variables share the same name as those input
    // variables on the state subaxis, and since we don't duplicate storage for variables with the
    // same name, they are essentially the same variable with FType::INPUT | FType::OUTPUT. During
    // the nonlinear solve, we have to iteratively update the guess (i.e., the input variables on
    // the state subaxis) until convergece. Once the nonlinear system has converged, the input
    // variables on the state subaxis _must_ contain the solution. Therefore, the output variables
    // _must_ also contain the solution upon convergence.

    // All that being said, if the result has AD graph, we need to propagate the graph to the output
    if (res.solution.requires_grad())
      assign_output(sol_assember.split_by_variable(res.solution));
  }

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    // IFT requires the Jacobian evaluated at the solution:
    _model.forward_maybe_jit(false, true, false);
    const auto jac_assembler = MatrixAssembler(_model.output_axis(), _model.input_axis());
    const auto J = jac_assembler.assemble_by_variable(_model.collect_output_derivatives());
    const auto derivs = jac_assembler.split_by_subaxis(J).at(RESIDUAL);
    const auto dr_ds = derivs.at(STATE);

    // Factorize the Jacobian once and for all
    const auto [LU, pivot] = linalg::lu_factor(dr_ds);

    // The actual IFT:
    for (const auto & [subaxis, deriv] : derivs)
    {
      if (subaxis == STATE)
        continue;
      const auto ift_assembler =
          MatrixAssembler(output_axis(), _model.input_axis().subaxis(subaxis));
      assign_output_derivatives(
          ift_assembler.split_by_variable(-linalg::lu_solve(LU, pivot, deriv)));
    }
  }
}
} // namespace neml2
