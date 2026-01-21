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
#include "neml2/equation_systems/HMatrix.h"

namespace neml2
{
register_NEML2_object(ImplicitUpdate);

OptionSet
ImplicitUpdate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      "Update an implicit model by solving the underlying nonlinear system of equations.";

  options.set<std::string>("implicit_model");
  options.set("implicit_model").doc() =
      "The model defining the nonlinear system of equations to be solved";

  options.set<std::string>("solver");
  options.set("solver").doc() = "Solver used to solve the nonlinear system of equations";

  // No jitting :/
  options.set<bool>("jit") = false;
  options.set("jit").suppressed() = true;

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _model(register_model("implicit_model")),
    _nl_sys(&_model),
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
ImplicitUpdate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  // The trial state is used as the initial guess
  const auto u0 = _nl_sys.u();

  // Solve for the next state
  const auto res = _solver->solve(_nl_sys, u0);
  _last_iterations = res.iterations;
  neml_assert(res.ret == NonlinearSolver::RetCode::SUCCESS, "Nonlinear solve failed.");

  if (out)
    assign_output(_nl_sys.umap(), res.solution);

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    // IFT requires the Jacobian evaluated at the solution:
    _model.forward_maybe_jit(false, true, false);
    const auto dr_du = _model.collect_output_derivatives(_nl_sys.bmap(), _nl_sys.umap());

    // collect dr/df
    std::vector<HMatrix> dr_dfs;
    dr_dfs.reserve(_nl_sys.unmap().size() + _nl_sys.gmap().size() + _nl_sys.gnmap().size());
    for (const auto & f : _nl_sys.unmap())
      dr_dfs.emplace_back(_model.collect_output_derivatives(_nl_sys.bmap(), {f}));
    for (const auto & f : _nl_sys.gmap())
      dr_dfs.emplace_back(_model.collect_output_derivatives(_nl_sys.bmap(), {f}));
    for (const auto & f : _nl_sys.gnmap())
      dr_dfs.emplace_back(_model.collect_output_derivatives(_nl_sys.bmap(), {f}));

    // du/df = - (dr/du)^{-1} (dr/df)
    std::vector<HMatrix> du_dfs;
    du_dfs.reserve(dr_dfs.size());
    // If the solver supports LU factorization, use that:
    if (_solver->linear_solver->support_lu_factorization())
    {
      auto [LU, pivot] = _solver->linear_solver->lu_factor(dr_du);
      for (const auto & dr_df : dr_dfs)
        du_dfs.emplace_back(-_solver->linear_solver->lu_solve(LU, pivot, dr_df));
    }
    // Otherwise, fall back to calling solve repeatedly
    else
      for (const auto & dr_df : dr_dfs)
        du_dfs.emplace_back(-_solver->linear_solver->solve(dr_du, dr_df));

    // assign derivatives back
    std::size_t i = 0;
    for (const auto & f : _nl_sys.unmap())
      assign_output_derivatives(_nl_sys.umap(), {f}, du_dfs[i++]);
    for (const auto & f : _nl_sys.gmap())
      assign_output_derivatives(_nl_sys.umap(), {f}, du_dfs[i++]);
    for (const auto & f : _nl_sys.gnmap())
      assign_output_derivatives(_nl_sys.umap(), {f}, du_dfs[i++]);
  }
}
} // namespace neml2
