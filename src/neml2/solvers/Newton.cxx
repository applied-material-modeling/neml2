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

#include "neml2/solvers/Newton.h"
#include <iomanip>
#include "neml2/misc/math.h"

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
Newton::solve(NonlinearSystem & system, const NonlinearSystem::SOL<false> & x0)
{
  // Solve always takes place in the "scaled" space
  auto x = system.scale(x0);

  // The initial residual for relative convergence check
  auto R = system.residual(x);
  auto nR = math::linalg::vector_norm(R);
  auto nR0 = nR.clone();

  // Check for initial convergence
  if (converged(0, nR0, nR0))
  {
    // The final update is only necessary if we use AD
    if (R.requires_grad())
      final_update(system, x, R, system.Jacobian<true>());

    return {RetCode::SUCCESS, system.unscale(x), 0};
  }

  // Prepare any solver internal data before the iterative update
  prepare(system, x);

  // Continuing iterating until one of:
  // 1. nR < atol (success)
  // 2. nR / nR0 < rtol (success)
  // 3. i > miters (failure)
  for (size_t i = 1; i < miters; i++)
  {
    auto J = system.Jacobian<true>();
    update(system, x, R, J);
    R = system.residual(x);
    nR = math::linalg::vector_norm(R);

    // Check for convergence
    if (converged(i, nR, nR0))
    {
      // The final update is only necessary if we use AD
      if (R.requires_grad())
        final_update(system, x, R, system.Jacobian<true>());

      return {RetCode::SUCCESS, system.unscale(x), i};
    }
  }

  return {RetCode::MAXITER, system.unscale(x), miters};
}

bool
Newton::converged(size_t itr, const torch::Tensor & nR, const torch::Tensor & nR0) const
{
  // LCOV_EXCL_START
  if (verbose)
    std::cout << "ITERATION " << std::setw(3) << itr << ", |R| = " << std::scientific
              << torch::max(nR).item<Real>() << ", |R0| = " << std::scientific
              << torch::max(nR0).item<Real>() << std::endl;
  // LCOV_EXCL_STOP

  return torch::all(torch::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>();
}

void
Newton::update(NonlinearSystem & /*system*/,
               NonlinearSystem::SOL<true> & x,
               const NonlinearSystem::RES<true> & r,
               const NonlinearSystem::JAC<true> & J)
{
  x = NonlinearSystem::SOL<true>(x.variable_data() + Tensor(solve_direction(r, J)));
}

void
Newton::final_update(NonlinearSystem & /*system*/,
                     NonlinearSystem::SOL<true> & x,
                     const NonlinearSystem::RES<true> & r,
                     const NonlinearSystem::JAC<true> & J)
{
  x = NonlinearSystem::SOL<true>(Tensor(x) + Tensor(solve_direction(r, J)));
}

NonlinearSystem::SOL<true>
Newton::solve_direction(const NonlinearSystem::RES<true> & r, const NonlinearSystem::JAC<true> & J)
{
  if (r.base_dim() == 0 && J.base_dim() == 0)
    return NonlinearSystem::SOL<true>(-Tensor(r) / Tensor(J));

  return NonlinearSystem::SOL<true>(-math::linalg::solve(J, r));
}

} // namespace neml2
