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

namespace neml2
{
OptionSet
NonlinearSolver::expected_options()
{
  OptionSet options = Solver::expected_options();

  options.set<double>("abs_tol") = 1e-10;
  options.set("abs_tol").doc() = "Absolute tolerance in the convergence criteria";

  options.set<double>("rel_tol") = 1e-8;
  options.set("rel_tol").doc() = "Relative tolerance in the convergence criteria";

  options.set<unsigned int>("max_its") = 100;
  options.set("max_its").doc() =
      "Maximum number of iterations allowed before issuing an error/exception";

  return options;
}

NonlinearSolver::NonlinearSolver(const OptionSet & options)
  : Solver(options),
    atol(options.get<double>("abs_tol")),
    rtol(options.get<double>("rel_tol")),
    miters(options.get<unsigned int>("max_its"))
{
}
} // namespace neml2
