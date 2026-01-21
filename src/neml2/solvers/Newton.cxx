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

#include <iostream>
#include <iomanip>

#include "neml2/solvers/Newton.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/equation_systems/HVector.h"
#include "neml2/equation_systems/HMatrix.h"

namespace neml2
{
register_NEML2_object(Newton);

OptionSet
Newton::expected_options()
{
  OptionSet options = NonlinearSolver::expected_options();
  options.doc() = "The standard Newton-Raphson solver which always takes the 'full' Newton step.";

  return options;
}

Newton::Newton(const OptionSet & options)
  : NonlinearSolver(options)
{
}

Newton::Result
Newton::solve(NonlinearSystem & system, const HVector & u0)
{
  auto u = u0;

  // The initial residual for relative convergence check
  system.set_u(u);
  auto b = system.b();
  auto nb = neml2::norm(b);
  auto nb0 = nb.clone();

  // Check for initial convergence
  if (converged(0, nb0, nb0))
  {
    // The final update is only necessary if we use AD
    if (b.requires_grad())
      final_update(system, u, b, system.A());

    return {RetCode::SUCCESS, u, 0};
  }

  // Prepare any solver internal data before the iterative update
  prepare(system, u);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    auto A = system.A();
    update(system, u, b, A);
    system.set_u(u);
    b = system.b();
    nb = neml2::norm(b);

    // Check for convergence
    if (converged(i, nb, nb0))
    {
      // The final update is only necessary if we use AD
      if (b.requires_grad())
        final_update(system, u, b, system.A());

      return {RetCode::SUCCESS, u, i};
    }
  }

  return {RetCode::MAXITER, u, miters};
}

bool
Newton::converged(size_t itr, const Scalar & nb, const Scalar & nb0) const
{
  // LCOV_EXCL_START
  if (verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << at::max(nb).item<double>() << ", |R0| = " << std::scientific
              << at::max(nb0).item<double>() << std::endl;
  // LCOV_EXCL_STOP

  return at::all(at::logical_or(nb < atol, nb / nb0 < rtol)).item<bool>();
}

void
Newton::update(NonlinearSystem & /*system*/, HVector & u, const HVector & b, const HMatrix & A)
{
  u.update_data(solve_direction(b, A));
}

void
Newton::final_update(NonlinearSystem & /*system*/,
                     HVector & u,
                     const HVector & b,
                     const HMatrix & A)
{
  u.update(solve_direction(b, A));
}

HVector
Newton::solve_direction(const HVector & b, const HMatrix & A)
{
  return linear_solver->solve(A, b);
}

} // namespace neml2
