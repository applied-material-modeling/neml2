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

#include "neml2/solvers/LinearSolver.h"
#include "neml2/base/Settings.h"

namespace neml2
{

OptionSet
LinearSolver::expected_options()
{
  OptionSet options = Solver::expected_options();
  return options;
}

LinearSolver::LinearSolver(const OptionSet & options)
  : Solver(options)
{
}

bool
LinearSolver::check_errors() const
{
  // at::linalg_solve_ex skips device synchronization unless check_errors is set to true. This is
  // much faster when the system of equations is well-conditioned, but it can lead to silent
  // failures when the system is ill-conditioned.
  //
  // We always set check_errors to true in debug mode, so that we can catch potential issues
  // without sacrificing performance in production.
  //
  // In non-debug modes, check_errors default to false. User can turn on error checking using
  // Settings/linalg_solve_check_errors=true.
#ifndef NDEBUG
  return true;
#else
  return settings().linalg_solve_check_errors();
#endif
}

}
