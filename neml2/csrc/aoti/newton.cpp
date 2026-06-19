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

#include "neml2/csrc/aoti/newton.h"
#include "neml2/csrc/aoti/nonlinear_system.h"
#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/assertions.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <ATen/ATen.h>

namespace neml2::aoti
{
namespace
{
// Bool convergence check: max-norm over the batch must clear (||b|| < atol)
// or (||b||/||b0|| < rtol). Single GPU->CPU sync via .item<bool>().
bool
check_converged(const at::Tensor & b_norm, const at::Tensor & b0_norm, double atol, double rtol)
{
  return at::all(at::logical_or(b_norm < atol, b_norm / b0_norm < rtol)).item<bool>();
}

// Tri-state stop check for the Newton loop. Folds the divergence detection
// (any NaN / Inf in the residual norm) into the same device → host sync as
// the existing convergence check, so on CUDA we still pay only one d2h per
// iteration. The extra cost is a single `isfinite` reduction kernel, which
// is negligible next to the step graph it runs alongside.
enum class StopStatus
{
  Continue = 0,
  Converged = 1,
  Diverged = 2,
};

inline StopStatus
check_stop(const at::Tensor & b_norm, const at::Tensor & b0_norm, double atol, double rtol)
{
  auto diverged = at::any(at::logical_not(at::isfinite(b_norm)));
  auto converged = at::all(at::logical_or(b_norm < atol, b_norm / b0_norm < rtol));
  // Stack on-device into a (2,) bool tensor and pull both bits over in one
  // d2h memcpy. The two reductions feed the same downstream sync, so this is
  // not strictly more sync than the original single-condition check.
  auto packed = at::stack({diverged, converged}).cpu();
  if (packed[0].item<bool>())
    return StopStatus::Diverged;
  if (packed[1].item<bool>())
    return StopStatus::Converged;
  return StopStatus::Continue;
}

// Number of trailing axes a per-group tensor carries beyond the dynamic-batch
// leading axes. BLOCK: sub_batch axes + 1 folded base axis. DENSE: 1.
int64_t
group_trail(const GroupLayout & g)
{
  return (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
}

// Per-group sum-of-squares reduction across all groups, summing each group's
// trailing (sub_batch + base) axes, leaving only the dynamic-batch leading
// axes. Used for the convergence norm and the Strong-Wolfe `b · du` scalar.
at::Tensor
pergroup_norm_sq(const std::vector<at::Tensor> & group_tensors,
                 const std::vector<GroupLayout> & groups)
{
  _assert(group_tensors.size() == groups.size(),
          "pergroup_norm_sq: tensor count ",
          group_tensors.size(),
          " != group count ",
          groups.size());
  at::Tensor acc;
  for (std::size_t k = 0; k < group_tensors.size(); ++k)
  {
    const auto & t = group_tensors[k];
    const int64_t trail = group_trail(groups[k]);
    _assert(
        t.dim() >= trail, "pergroup_norm_sq: group tensor ndim=", t.dim(), " < trail ndim=", trail);
    auto sq = t * t;
    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(trail));
    for (int64_t d = t.dim() - trail; d < t.dim(); ++d)
      dims.push_back(d);
    sq = sq.sum(dims);
    acc = acc.defined() ? (acc + sq) : sq;
  }
  return acc;
}

// Pairwise sum: sum_k sum_{non-batch}(a_groups[k] * b_groups[k]). Both vectors
// must be the same length with shape-matching entries. Used for the Armijo
// line-search `b · du` criterion.
at::Tensor
pergroup_dot(const std::vector<at::Tensor> & a_groups,
             const std::vector<at::Tensor> & b_groups,
             const std::vector<GroupLayout> & groups)
{
  _assert(a_groups.size() == b_groups.size() && a_groups.size() == groups.size(),
          "pergroup_dot: vector size mismatch (",
          a_groups.size(),
          "/",
          b_groups.size(),
          " vs ",
          groups.size(),
          ")");
  at::Tensor acc;
  for (std::size_t k = 0; k < a_groups.size(); ++k)
  {
    const int64_t trail = group_trail(groups[k]);
    auto prod = a_groups[k] * b_groups[k];
    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(trail));
    for (int64_t d = a_groups[k].dim() - trail; d < a_groups[k].dim(); ++d)
      dims.push_back(d);
    prod = prod.sum(dims);
    acc = acc.defined() ? (acc + prod) : prod;
  }
  return acc;
}

// Reshape `(*B,)` alpha-per-batch to broadcast against a per-group tensor by
// appending `group_trail` trailing size-1 axes.
at::Tensor
alpha_for_group(const at::Tensor & alpha, const GroupLayout & g)
{
  auto out = alpha;
  const int64_t trail = group_trail(g);
  for (int64_t k = 0; k < trail; ++k)
    out = out.unsqueeze(-1);
  return out;
}
} // namespace

Newton::Newton(SolverConfig cfg)
  : _cfg(std::move(cfg))
{
}

NewtonResult
Newton::solve(const NonlinearSystem & sys, const std::vector<at::Tensor> & u0) const
{
  const auto & unknown_layout = sys.unknown_layout();
  const auto & residual_layout = sys.residual_layout();

  _assert(u0.size() == unknown_layout.size(),
          "Newton::solve: u0 group count ",
          u0.size(),
          " != unknown group count ",
          unknown_layout.size());

  // Per-group u, kept at natural per-group shape (BLOCK: (*B, *group_sub,
  // group_base_total); DENSE: (*B, group_total)). The Newton loop updates each
  // group independently via `u[i] = u[i] + alpha * du[i]`, where alpha is
  // broadcast-reshaped per group's trailing axes by `alpha_for_group`.
  std::vector<at::Tensor> u;
  u.reserve(u0.size());
  for (const auto & t : u0)
    u.push_back(t.contiguous());

  // Initial per-residual-group b vector.
  auto b_outs = sys.residual(u);
  _assert(b_outs.size() == residual_layout.size(),
          "Newton::solve: residual() returned ",
          b_outs.size(),
          " tensors, expected one per residual group (",
          residual_layout.size(),
          ")");
  // Per-batch norm of the residual: sqrt of sum across residual groups of sum
  // over (group_sub + group_base) of group^2. Leading axes are the
  // dynamic-batch shape.
  auto b0_norm_sq = pergroup_norm_sq(b_outs, residual_layout);
  auto b0_norm = b0_norm_sq.sqrt();

  const char * trace_env = std::getenv("NEML2_AOTI_TRACE_NEWTON");
  const int trace_level = trace_env ? std::atoi(trace_env) : 0;
  const bool trace = trace_level >= 1;
  const bool verbose = trace_level >= 2;

  if (check_converged(b0_norm, b0_norm, _cfg.atol, _cfg.rtol))
  {
    if (trace)
      std::cerr << "[aoti newton]iters=0 (converged at predictor) "
                << "b0_norm=" << b0_norm.max().item<double>() << std::endl;
    return {std::move(u), /*converged=*/true, /*iterations=*/0};
  }
  std::size_t reached = _cfg.miters;
  if (verbose)
    std::cerr << "ITERATION " << std::setw(3) << 0 << ", |R| = " << std::scientific
              << b0_norm.max().item<double>() << ", |R0| = " << std::scientific
              << b0_norm.max().item<double>() << std::endl;
  for (std::size_t i = 1; i < _cfg.miters; ++i)
  {
    // step() returns (du_groups, b_curr_groups). We already have b_outs from
    // the prior iteration, so b_curr is discarded (kept by the interface for
    // symmetry / future use).
    auto step_result = sys.step(u);
    std::vector<at::Tensor> & du = step_result.first;
    _assert(du.size() == unknown_layout.size(),
            "Newton::solve: step() returned ",
            du.size(),
            " du tensors, expected one per unknown group (",
            unknown_layout.size(),
            ")");

    auto alpha = at::ones_like(b0_norm);
    // Armijo `b · du`: dot product across all residual+unknown groups. For a
    // square system, residual groups align with unknown groups by position
    // (residual_layout[k] is the residual block for unknown_layout[k]), with
    // matching per-group structure so trailing axes match per pair.
    const auto b_dot_du = pergroup_dot(b_outs, du, unknown_layout);
    const auto nb_curr_sq = pergroup_norm_sq(b_outs, residual_layout);

    std::vector<at::Tensor> u_trial(unknown_layout.size());
    std::vector<at::Tensor> b_trial;
    if (_cfg.ls_max_iters <= 1)
    {
      for (std::size_t k = 0; k < unknown_layout.size(); ++k)
        u_trial[k] = (u[k] + du[k]).contiguous();
      b_trial = sys.residual(u_trial);
      _assert(b_trial.size() == residual_layout.size(),
              "Newton::solve: residual() returned ",
              b_trial.size(),
              " tensors, expected one per residual group");
    }
    else
    {
      for (std::size_t k_ls = 1; k_ls < _cfg.ls_max_iters; ++k_ls)
      {
        for (std::size_t k = 0; k < unknown_layout.size(); ++k)
        {
          const auto alpha_b = alpha_for_group(alpha, unknown_layout[k]);
          u_trial[k] = (u[k] + alpha_b * du[k]).contiguous();
        }
        b_trial = sys.residual(u_trial);
        _assert(b_trial.size() == residual_layout.size(),
                "Newton::solve: residual() returned ",
                b_trial.size(),
                " tensors, expected one per residual group");

        const auto nb_trial_sq = pergroup_norm_sq(b_trial, residual_layout);
        at::Tensor crit;
        if (_cfg.ls_type == "STRONG_WOLFE")
          crit = (1.0 - _cfg.ls_c * alpha) * nb_curr_sq;
        else // BACKTRACKING
          crit = nb_curr_sq - 2.0 * _cfg.ls_c * alpha * b_dot_du;

        if (verbose)
          std::cerr << "     LS ITERATION " << std::setw(3) << k_ls
                    << ", min(alpha) = " << std::scientific << alpha.min().item<double>()
                    << ", max(||R||) = " << std::scientific
                    << nb_trial_sq.sqrt().max().item<double>()
                    << ", min(||Rc||) = " << std::scientific << crit.sqrt().min().item<double>()
                    << std::endl;

        const auto stop = at::logical_or(nb_trial_sq <= crit, nb_trial_sq <= _cfg.atol * _cfg.atol);
        if (stop.all().item<bool>())
          break;
        alpha = at::where(stop, alpha, alpha / _cfg.ls_cutback);
      }
    }

    u = std::move(u_trial);
    b_outs = std::move(b_trial);

    const auto b_norm_sq = pergroup_norm_sq(b_outs, residual_layout);
    const auto b_norm = b_norm_sq.sqrt();
    if (verbose)
      std::cerr << "ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << b_norm.max().item<double>() << ", |R0| = " << std::scientific
                << b0_norm.max().item<double>() << std::endl;
    const auto status = check_stop(b_norm, b0_norm, _cfg.atol, _cfg.rtol);
    if (status == StopStatus::Diverged)
      throw ConvergenceError("AOTI Newton diverged at iter " + std::to_string(i) +
                             " (non-finite residual). Consider tightening the predictor, "
                             "increasing max_linesearch_iterations, or reducing the time step.");
    if (status == StopStatus::Converged)
    {
      reached = i;
      if (trace)
        std::cerr << "[aoti newton]iters=" << i << " (converged) "
                  << "b0_norm=" << b0_norm.max().item<double>()
                  << " b_norm=" << b_norm.max().item<double>() << std::endl;
      return {std::move(u), /*converged=*/true, /*iterations=*/i};
    }
  }

  const auto final_norm = pergroup_norm_sq(b_outs, residual_layout).sqrt();
  if (trace)
    std::cerr << "[aoti newton]iters=" << reached
              << " (MAXITERS HIT) b0_norm=" << b0_norm.max().item<double>()
              << " b_norm=" << final_norm.max().item<double>() << std::endl;

  // Maxed out without converging. A non-converged iterate is not a usable
  // solution, so this is a *recoverable* failure (ConvergenceError): a
  // time-stepping consumer can cut the step and retry. Convergence is an
  // all-reduce over the batch, so this fires if *any* batch member is still
  // unconverged. (Divergence above throws the same type for the same reason.)
  throw ConvergenceError(
      "AOTI Newton failed to converge in " + std::to_string(_cfg.miters) +
      " iterations (max |R| = " + std::to_string(final_norm.max().item<double>()) +
      ", |R0| = " + std::to_string(b0_norm.max().item<double>()) +
      ", atol = " + std::to_string(_cfg.atol) + ", rtol = " + std::to_string(_cfg.rtol) +
      "). Consider reducing the time step or raising max_iterations.");
}
} // namespace neml2::aoti
