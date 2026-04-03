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

#include "neml2/solvers/NonlinearSolver.h"
#include "neml2/solvers/LinearSolver.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
OptionSet
NonlinearSolver::expected_options()
{
  OptionSet options = Solver::expected_options();

  options.add<std::string>("linear_solver", "The linear solver to use within the nonlinear solver");

  options.add<double>("abs_tol", 1e-10, "Absolute tolerance in the convergence criteria");
  options.add<double>("rel_tol", 1e-8, "Relative tolerance in the convergence criteria");
  options.add<unsigned int>(
      "max_its", 100, "Maximum number of iterations allowed before issuing an error/exception");

  return options;
}

NonlinearSolver::NonlinearSolver(const OptionSet & options)
  : Solver(options),
    atol(options.get<double>("abs_tol")),
    rtol(options.get<double>("rel_tol")),
    miters(options.get<unsigned int>("max_its"))
{
  neml_assert(
      options.user_specified("linear_solver"),
      "The 'linear_solver' option is required. Refer to "
      "https://applied-material-modeling.github.io/neml2/migration-200-210.html#linear_solver for "
      "more information.");
  linear_solver = get_solver<LinearSolver>("linear_solver");
}

void
NonlinearSolver::to(const TensorOptions & options)
{
  linear_solver->to(options);
}

} // namespace neml2
