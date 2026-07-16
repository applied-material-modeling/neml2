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

// Standalone numerics test for the `substep_del_tol` step-norm gate in
// `Newton::solve_masked` (newton.cpp). It drives a hand-built scalar cubic
// `NonlinearSystem` directly -- no AOTI artifact, no metadata round-trip -- so
// the gate logic is exercised in isolation and deterministically.
//
// The gate: a row is accepted via the relative branch (||b||/||b0|| < rtol) only
// if additionally ||du|| < substep_del_tol*(||u|| + atol). The cubic residual
// r(u) = u^3 (root u* = 0) started from a LARGE u0 inflates ||b0|| = u0^3 by ~10
// orders of magnitude, so rtol*||b0|| becomes a loose *absolute* floor. Newton
// converges geometrically in u (u -> 2u/3) but the residual drops below that
// floor while u is still O(1) -- a non-physical, still-moving waypoint.
//
//   * LOOSE substep_del_tol (pure relative test): the masked solve certifies the
//     waypoint as converged and freezes a garbage iterate far from the root.
//   * DEFAULT (tight) substep_del_tol: the step-norm gate rejects the waypoint
//     (the committed |du| ~ u/3 is nowhere near vanishing), so Newton keeps going
//     and reaches the true root via the absolute branch.
//
// No line-search dependence: with ls_max_iters = 1 the full Newton step is taken
// each iteration, so the u -> 2u/3 trajectory is exact.

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/newton.h"
#include "neml2/csrc/aoti/nonlinear_system.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
// Scalar cubic residual r(u) = u^3, root u* = 0, as a single DENSE group. Solver
// conventions (see NonlinearSystem): residual() returns b = -r; step() returns
// (du, b) solving A du = b with A = dr/du = 3u^2, i.e. du = -u/3. `b_outs` is
// always recomputed from residual() inside newton_iterate, so step()'s returned
// b is only for interface symmetry.
class CubicSystem : public NonlinearSystem
{
public:
  CubicSystem()
    : _layout{GroupLayout{"dense", {}}}
  {
  }

  std::vector<at::Tensor> residual(const std::vector<at::Tensor> & u) const override
  {
    return {-(u[0] * u[0] * u[0])}; // b = -r = -u^3
  }

  std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>>
  step(const std::vector<at::Tensor> & u) const override
  {
    std::vector<at::Tensor> du{-u[0] / 3.0}; // A^{-1} b = (-u^3)/(3u^2) = -u/3
    return {std::move(du), residual(u)};
  }

  const std::vector<GroupLayout> & unknown_layout() const override { return _layout; }
  const std::vector<GroupLayout> & residual_layout() const override { return _layout; }

private:
  std::vector<GroupLayout> _layout;
};

SolverConfig
make_cfg(double substep_del_tol)
{
  SolverConfig cfg;
  cfg.atol = 1.0e-10;
  cfg.rtol = 1.0e-8;
  cfg.miters = 60; // enough for the tight gate to grind u0=2154 down to the root
  cfg.ls_type = "BACKTRACKING";
  cfg.ls_max_iters = 1; // full Newton step (no damping) -> exact u -> 2u/3
  cfg.ls_cutback = 2.0;
  cfg.ls_c = 1.0e-4;
  cfg.substep_del_tol = substep_del_tol;
  return cfg;
}
} // namespace

int
main()
{
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  CubicSystem sys;

  // Single DENSE scalar group, dynamic batch (1,): shape (B=1, group_total=1).
  // u0 large -> ||b0|| = u0^3 ~ 1e10 so rtol*||b0|| ~ 1e2 is a loose floor.
  const auto u0 = at::full({1, 1}, 2154.0, opts);

  // Loose gate == pure relative test: freezes a mid-trajectory waypoint.
  const auto res_loose = Newton(make_cfg(1.0e30)).solve_masked(sys, {u0});
  const double u_loose = res_loose.u[0].abs().max().item<double>();

  // Tight gate (the shipped default): the step-norm gate rejects the waypoint,
  // so Newton continues to the physical root u* = 0.
  const auto res_tight = Newton(make_cfg(1.0e-6)).solve_masked(sys, {u0});
  const double u_tight = res_tight.u[0].abs().max().item<double>();

  std::printf("substep_del_tol gate: u_loose=%.6e  u_tight=%.6e\n", u_loose, u_tight);

  // Both solves report convergence -- the loose one wrongly (frozen waypoint),
  // the tight one truly (the root).
  NEML2_CHECK(res_loose.converged);
  NEML2_CHECK(res_tight.converged);
  // Loose gate froze a non-physical waypoint far from the root.
  NEML2_CHECK(u_loose > 1.0);
  // Tight gate reached the true root.
  NEML2_CHECK(u_tight < 1.0e-3);
  // The gate materially changed the converged result.
  NEML2_CHECK(std::fabs(u_loose - u_tight) > 1.0);

  std::printf("OK\n");
  return 0;
}
