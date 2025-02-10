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

#include <iomanip>

#include "neml2/solvers/NewtonWithTrustRegion.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/where.h"
#include "neml2/tensors/functions/bmv.h"
#include "neml2/tensors/functions/bvv.h"
#include "neml2/tensors/functions/linalg/vector_norm.h"

namespace neml2
{
register_NEML2_object(NewtonWithTrustRegion);

OptionSet
NewtonWithTrustRegion::expected_options()
{
  OptionSet options = Newton::expected_options();
  options.doc() =
      "A trust-region Newton solver. The step size and direction are modified by solving a "
      "constrained minimization problem using the local quadratic approximation. The default "
      "solver parameters are chosen based on a limited set of problems we tested on and are "
      "expected to be tuned.";

  options.set<Real>("delta_0") = 1.0;
  options.set("delta_0").doc() = "Initial trust region radius";

  options.set<Real>("delta_max") = 10.0;
  options.set("delta_max").doc() = "Maximum trust region radius";

  options.set<Real>("reduce_criteria") = 0.25;
  options.set("reduce_criteria").doc() = "The trust region radius is reduced when the merit "
                                         "function reduction is below this threshold";

  options.set<Real>("expand_criteria") = 0.75;
  options.set("expand_criteria").doc() = "The trust region radius is increased when the merit "
                                         "function reduction is above this threshold";

  options.set<Real>("reduce_factor") = 0.25;
  options.set("reduce_factor").doc() = "Factor to apply when reducing the trust region radius";

  options.set<Real>("expand_factor") = 2.0;
  options.set("expand_factor").doc() = "Factor to apply when increasing the trust region radius";

  options.set<Real>("accept_criteria") = 0.1;
  options.set("accept_criteria").doc() =
      "Reject the current step when the merit function reduction is below this threshold";

  options.set<Real>("subproblem_rel_tol") = 1e-6;
  options.set("subproblem_rel_tol").doc() = "Relative tolerance used for the quadratic sub-problem";

  options.set<Real>("subproblem_abs_tol") = 1e-8;
  options.set("subproblem_abs_tol").doc() = "Absolute tolerance used for the quadratic sub-problem";

  options.set<unsigned int>("subproblem_max_its") = 10;
  options.set("subproblem_max_its").doc() =
      "Maximum number of allowable iterations when solving the quadratic sub-problem";

  return options;
}

NewtonWithTrustRegion::NewtonWithTrustRegion(const OptionSet & options)
  : Newton(options),
    _subproblem(subproblem_options(options)),
    _subproblem_solver(subproblem_solver_options(options)),
    _delta_0(options.get<Real>("delta_0")),
    _delta_max(options.get<Real>("delta_max")),
    _reduce_criteria(options.get<Real>("reduce_criteria")),
    _expand_criteria(options.get<Real>("expand_criteria")),
    _reduce_factor(options.get<Real>("reduce_factor")),
    _expand_factor(options.get<Real>("expand_factor")),
    _accept_criteria(options.get<Real>("accept_criteria"))
{
}

OptionSet
NewtonWithTrustRegion::subproblem_options(const OptionSet & /*options*/) const
{
  // By default the nonlinear system turns off automatic scaling (which is what we want here)
  return TrustRegionSubProblem::expected_options();
}

OptionSet
NewtonWithTrustRegion::subproblem_solver_options(const OptionSet & options) const
{
  auto solver_options = Newton::expected_options();
  solver_options.set<Real>("abs_tol") = options.get<Real>("subproblem_abs_tol");
  solver_options.set<Real>("rel_tol") = options.get<Real>("subproblem_rel_tol");
  solver_options.set<unsigned int>("max_its") = options.get<unsigned int>("subproblem_max_its");
  return solver_options;
}

void
NewtonWithTrustRegion::prepare(const NonlinearSystem & /*system*/,
                               const NonlinearSystem::Sol<true> & x)
{
  _delta = Scalar::full(x.batch_sizes(), _delta_0, x.options());
}

void
NewtonWithTrustRegion::update(NonlinearSystem & system,
                              NonlinearSystem::Sol<true> & x,
                              const NonlinearSystem::Res<true> & r,
                              const NonlinearSystem::Jac<true> & J)
{
  auto p = solve_direction(r, J);

  // Predicted reduction in the merit function
  auto nr = linalg::vector_norm(r);
  auto red_b = merit_function_reduction(r, J, p);

  // Actual reduction in the objective function
  NonlinearSystem::Sol<true> xp(Tensor(x) + Tensor(p));
  auto rp = system.residual(xp);
  auto nrp = linalg::vector_norm(rp);
  auto red_a = 0.5 * pow(nr, 2.0) - 0.5 * pow(nrp, 2.0);

  // Quality of the subproblem solution compared to the quadratic model
  auto rho = red_a / red_b;

  // Adjust the trust region based on the quality of the subproblem
  _delta.batch_index_put_({rho < _reduce_criteria},
                          _reduce_factor * _delta.batch_index({rho < _reduce_criteria}));
  _delta.batch_index_put_({rho > _expand_criteria},
                          at::clamp(_expand_factor * _delta.batch_index({rho > _expand_criteria}),
                                    c10::nullopt,
                                    _delta_max));

  // Accept or reject the current step
  auto accept = (rho >= _accept_criteria).unsqueeze(-1);

  // Do some printing if verbose
  if (verbose)
  {
    std::cout << "     RHO MIN/MAX            : " << std::scientific << at::min(rho).item<Real>()
              << "/" << std::scientific << at::max(rho).item<Real>() << std::endl;
    std::cout << "     ACCEPTANCE RATE        : " << at::sum(accept).item<Size>() << "/"
              << utils::storage_size(_delta.batch_sizes().concrete()) << std::endl;
    std::cout << "     ADJUSTED DELTA MIN/MAX : " << std::scientific << at::min(_delta).item<Real>()
              << "/" << std::scientific << at::max(_delta).item<Real>() << std::endl;
  }

  x = NonlinearSystem::Sol<true>(neml2::where(accept, Tensor(xp), x.variable_data()));
}

NonlinearSystem::Sol<true>
NewtonWithTrustRegion::solve_direction(const NonlinearSystem::Res<true> & r,
                                       const NonlinearSystem::Jac<true> & J)
{
  // The full Newton step
  auto p_newton = Newton::solve_direction(r, J);

  // The trust region step (obtained by solving the bound constrained subproblem)
  _subproblem.reinit(r, J, _delta);
  auto res = _subproblem_solver.solve(_subproblem,
                                      NonlinearSystem::Sol<false>(Tensor::zeros_like(_delta)));

  // Do some printing if verbose
  if (verbose)
  {
    std::cout << "     TRUST-REGION ITERATIONS: " << res.iterations << std::endl;
    std::cout << "     ACTIVE CONSTRAINTS     : " << at::sum(res.solution > 0).item<Size>() << "/"
              << utils::storage_size(res.solution.batch_sizes().concrete()) << std::endl;
  }

  auto s = Scalar(at::clamp(res.solution, 0.0), _delta.batch_sizes());
  auto p_trust = -_subproblem.preconditioned_direction(s);

  // Now select between the two... Basically take the full Newton step whenever possible
  auto newton_inside_trust_region =
      (linalg::vector_norm(p_newton) <= sqrt(2.0 * _delta)).unsqueeze(-1);

  return NonlinearSystem::Sol<true>(
      Tensor(at::where(newton_inside_trust_region, p_newton, p_trust), p_newton.batch_sizes()));
}

Scalar
NewtonWithTrustRegion::merit_function_reduction(const NonlinearSystem::Res<true> & r,
                                                const NonlinearSystem::Jac<true> & J,
                                                const NonlinearSystem::Sol<true> & p) const
{
  auto Jp = bmv(J, p);
  return -bvv(r, Jp) - 0.5 * bvv(Jp, Jp);
}

} // namespace neml2
