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

// ----------------------------------------------------------------------------
// Value / Newton path: forward-segment runner + the per-group implicit solve
// ----------------------------------------------------------------------------

#include "neml2/csrc/aoti/internal.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

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
} // namespace

void
Model::Impl::_run_forward_segment(const Segment & seg,
                                  std::map<std::string, at::Tensor> & state) const
{
  std::vector<at::Tensor> inputs;
  inputs.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
  {
    auto it = state.find(name);
    _assert(it != state.end(),
            "aoti::Model: forward segment needs input '",
            name,
            "' which is not in the state map.");
    inputs.push_back(it->second.contiguous());
  }
  // Promoted-parameter tail.
  for (auto & p : _gather_params(seg.param_inputs))
    inputs.push_back(std::move(p));

  const auto outs = seg.fwd_loader->run(inputs);
  _assert(outs.size() == seg.fwd_outputs.size(),
          "aoti::Model: forward segment returned ",
          outs.size(),
          " tensors, expected ",
          seg.fwd_outputs.size());
  for (std::size_t i = 0; i < seg.fwd_outputs.size(); ++i)
    state[seg.fwd_outputs[i]] = outs[i];
}

std::vector<at::Tensor>
Model::Impl::_pack_groups(const std::map<std::string, at::Tensor> & state,
                          const std::vector<Segment::GroupInfo> & groups) const
{
  // Mirrors :meth:`AssembledVector.from_dict` -- for each group, cat
  // per-var contributions along the last axis. BLOCK groups keep the
  // group's sub_batch axes (each var's sub matches the group's by
  // construction); DENSE groups fold each var's sub into base then cat.
  std::vector<at::Tensor> packed;
  packed.reserve(groups.size());
  for (const auto & group : groups)
  {
    std::vector<at::Tensor> parts;
    parts.reserve(group.per_var_info.size());
    for (const auto & v : group.per_var_info)
    {
      auto it = state.find(v.name);
      _assert(
          it != state.end(), "aoti::Model::_pack_groups: state missing variable '", v.name, "'.");
      const auto & t = it->second;
      const int64_t var_trail =
          static_cast<int64_t>(v.sub_batch_shape.size() + v.base_shape.size());
      _assert(t.dim() >= var_trail,
              "aoti::Model::_pack_groups: variable '",
              v.name,
              "' ndim=",
              t.dim(),
              " < sub_batch+base ndim=",
              var_trail);
      int64_t base_total = 1;
      for (auto s : v.base_shape)
        base_total *= s;
      int64_t sub_total = 1;
      for (auto s : v.sub_batch_shape)
        sub_total *= s;
      const int64_t dyn_ndim = t.dim() - var_trail;
      std::vector<int64_t> target;
      target.reserve(static_cast<std::size_t>(dyn_ndim) + v.sub_batch_shape.size() + 1);
      for (int64_t d = 0; d < dyn_ndim; ++d)
        target.push_back(t.size(d));
      if (group.structure == "block")
      {
        for (auto s : v.sub_batch_shape)
          target.push_back(s);
        target.push_back(base_total);
      }
      else
      {
        target.push_back(sub_total * base_total);
      }
      parts.push_back(t.contiguous().reshape(target));
    }
    _assert(!parts.empty(),
            "aoti::Model::_pack_groups: encountered an empty group; v7 layouts "
            "always have at least one variable per declared group.");
    packed.push_back(at::cat(parts, /*dim=*/-1).contiguous());
  }
  return packed;
}

void
Model::Impl::_unpack_groups(const std::vector<at::Tensor> & group_tensors,
                            const std::vector<Segment::GroupInfo> & groups,
                            std::map<std::string, at::Tensor> & state) const
{
  _assert(group_tensors.size() == groups.size(),
          "aoti::Model::_unpack_groups: tensor count ",
          group_tensors.size(),
          " != group count ",
          groups.size());
  for (std::size_t gi = 0; gi < groups.size(); ++gi)
  {
    const auto & gt = group_tensors[gi];
    const auto & group = groups[gi];
    // BLOCK group tensor: (*B, *group_sub, group_base_total).
    // DENSE group tensor: (*B, group_total).
    const int64_t group_trail =
        (group.structure == "block") ? static_cast<int64_t>(group.sub_batch_shape.size()) + 1 : 1;
    _assert(gt.dim() >= group_trail,
            "aoti::Model::_unpack_groups: group tensor ndim=",
            gt.dim(),
            " < trail ndim=",
            group_trail);
    const int64_t dyn_ndim = gt.dim() - group_trail;
    std::vector<int64_t> batch_shape;
    batch_shape.reserve(static_cast<std::size_t>(dyn_ndim));
    for (int64_t d = 0; d < dyn_ndim; ++d)
      batch_shape.push_back(gt.size(d));

    int64_t offset = 0;
    for (const auto & v : group.per_var_info)
    {
      int64_t base_total = 1;
      for (auto s : v.base_shape)
        base_total *= s;
      int64_t sub_total = 1;
      for (auto s : v.sub_batch_shape)
        sub_total *= s;
      const int64_t segment_size =
          (group.structure == "block") ? base_total : (sub_total * base_total);
      auto part = gt.narrow(/*dim=*/-1, /*start=*/offset, /*length=*/segment_size);

      std::vector<int64_t> target = batch_shape;
      for (auto s : v.sub_batch_shape)
        target.push_back(s);
      for (auto s : v.base_shape)
        target.push_back(s);
      state[v.name] = part.reshape(target).contiguous();
      offset += segment_size;
    }
  }
}

void
Model::Impl::_run_implicit_segment(const Segment & seg,
                                   std::map<std::string, at::Tensor> & state,
                                   std::vector<at::Tensor> & u_solved_groups,
                                   std::vector<at::Tensor> & g_groups) const
{
  // Per-variable predictor (optional): runs BEFORE we pack, since
  // predictor output is per-variable and lands in state[u.name].
  // Initial per-variable u defaults to zero at natural shape; the
  // predictor overrides matching names.
  _assert(!seg.givens.empty(),
          "aoti::Model: implicit segment has no givens; cannot infer batch shape.");
  auto it_g0 = state.find(seg.givens[0].name);
  _assert(it_g0 != state.end(),
          "aoti::Model: implicit segment needs given '",
          seg.givens[0].name,
          "' which is not in the state map.");
  const auto opts = it_g0->second.options();
  const auto g0 = it_g0->second;
  const auto & g0_info = seg.givens[0];
  const int64_t g0_trail =
      static_cast<int64_t>(g0_info.sub_batch_shape.size() + g0_info.base_shape.size());
  _assert(g0.dim() >= g0_trail,
          "aoti::Model: given tensor '",
          g0_info.name,
          "' ndim=",
          g0.dim(),
          " < declared sub_batch_ndim+base_ndim=",
          g0_trail);
  std::vector<int64_t> batch_shape(g0.sizes().begin(), g0.sizes().end() - g0_trail);

  auto _full_shape = [&](const Segment::VarInfo & v) -> std::vector<int64_t>
  {
    std::vector<int64_t> shape(batch_shape);
    shape.insert(shape.end(), v.sub_batch_shape.begin(), v.sub_batch_shape.end());
    shape.insert(shape.end(), v.base_shape.begin(), v.base_shape.end());
    return shape;
  };

  // Seed per-variable zero unknowns into state so the pack picks them up.
  for (const auto & v : seg.unknowns)
    state[v.name] = at::zeros(_full_shape(v), opts);

  if (seg.predictor_loader)
  {
    std::vector<at::Tensor> p_inputs;
    p_inputs.reserve(seg.predictor_inputs.size() + seg.param_inputs.size());
    for (const auto & name : seg.predictor_inputs)
    {
      auto it = state.find(name);
      _assert(it != state.end(),
              "aoti::Model: implicit segment predictor needs input '",
              name,
              "' which is not in the state map.");
      p_inputs.push_back(it->second.contiguous());
    }
    for (auto & p : _gather_params(seg.param_inputs))
      p_inputs.push_back(std::move(p));

    const auto p_outs = seg.predictor_loader->run(p_inputs);
    _assert(p_outs.size() == seg.predictor_outputs.size(),
            "aoti::Model: predictor returned ",
            p_outs.size(),
            " outputs, expected ",
            seg.predictor_outputs.size());

    // Route predictor outputs to the matching unknown slot by name.
    for (std::size_t i = 0; i < p_outs.size(); ++i)
      state[seg.predictor_outputs[i]] = p_outs[i].to(opts).contiguous();
  }

  // Pack per-group inputs at solve start (one at::cat per group).
  g_groups = _pack_groups(state, seg.given_groups);
  auto u0_groups = _pack_groups(state, seg.unknown_groups);

  // Newton solve (per-group).
  u_solved_groups = _solve_newton(seg, u0_groups, g_groups);

  // Unpack converged per-group unknowns back to per-variable state for
  // downstream forward segments / master outputs to read by name.
  _unpack_groups(u_solved_groups, seg.unknown_groups, state);
}

at::Tensor
Model::Impl::_pergroup_norm_sq(const std::vector<at::Tensor> & group_tensors,
                               const std::vector<Segment::GroupInfo> & groups)
{
  _assert(group_tensors.size() == groups.size(),
          "_pergroup_norm_sq: tensor count ",
          group_tensors.size(),
          " != group count ",
          groups.size());
  at::Tensor acc;
  for (std::size_t k = 0; k < group_tensors.size(); ++k)
  {
    const auto & t = group_tensors[k];
    const auto & g = groups[k];
    // BLOCK group tensor: (*B, *group_sub, group_base_total) -> reduce
    // sub.size()+1 trailing axes. DENSE: (*B, group_total) -> reduce 1.
    const int64_t trail =
        (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
    _assert(t.dim() >= trail,
            "_pergroup_norm_sq: group tensor ndim=",
            t.dim(),
            " < trail ndim=",
            trail);
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

at::Tensor
Model::Impl::_pergroup_dot(const std::vector<at::Tensor> & a_groups,
                           const std::vector<at::Tensor> & b_groups,
                           const std::vector<Segment::GroupInfo> & groups)
{
  _assert(a_groups.size() == b_groups.size() && a_groups.size() == groups.size(),
          "_pergroup_dot: vector size mismatch (",
          a_groups.size(),
          "/",
          b_groups.size(),
          " vs ",
          groups.size(),
          ")");
  at::Tensor acc;
  for (std::size_t k = 0; k < a_groups.size(); ++k)
  {
    const auto & g = groups[k];
    const int64_t trail =
        (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
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

at::Tensor
Model::Impl::_alpha_for_group(const at::Tensor & alpha, const Segment::GroupInfo & g)
{
  // BLOCK group tensor: (*B, *group_sub, group_base_total) -> append
  // sub.size()+1 trailing size-1 axes. DENSE: append 1.
  auto out = alpha;
  const int64_t trail =
      (g.structure == "block") ? static_cast<int64_t>(g.sub_batch_shape.size()) + 1 : 1;
  for (int64_t k = 0; k < trail; ++k)
    out = out.unsqueeze(-1);
  return out;
}

std::vector<at::Tensor>
Model::Impl::_solve_newton(const Segment & seg,
                           const std::vector<at::Tensor> & u0_groups,
                           const std::vector<at::Tensor> & g_groups) const
{
  _assert(u0_groups.size() == seg.unknown_groups.size(),
          "_solve_newton: u0 group count ",
          u0_groups.size(),
          " != unknown_groups ",
          seg.unknown_groups.size());
  _assert(g_groups.size() == seg.given_groups.size(),
          "_solve_newton: g group count ",
          g_groups.size(),
          " != given_groups ",
          seg.given_groups.size());

  // Per-group u, kept at natural per-group shape (BLOCK: (*B, *group_sub,
  // group_base_total); DENSE: (*B, group_total)). The Newton loop
  // updates each group independently via `u_groups[i] = u_groups[i] +
  // alpha * du_groups[i]`, where alpha is broadcast-reshaped per
  // group's trailing axes by `_alpha_for_group`.
  std::vector<at::Tensor> u;
  u.reserve(u0_groups.size());
  for (const auto & t : u0_groups)
    u.push_back(t.contiguous());

  // Given groups stay constant throughout the solve.
  std::vector<at::Tensor> g;
  g.reserve(g_groups.size());
  for (const auto & t : g_groups)
    g.push_back(t.contiguous());

  const auto pre_tail = _gather_params(seg.param_inputs);

  // Build a loader-call input list: (*u_groups, *g_groups, *params).
  auto build_inputs = [&](const std::vector<at::Tensor> & u_pack)
  {
    std::vector<at::Tensor> v;
    v.reserve(u_pack.size() + g.size() + pre_tail.size());
    for (const auto & t : u_pack)
      v.push_back(t.contiguous());
    for (const auto & t : g)
      v.push_back(t);
    for (const auto & t : pre_tail)
      v.push_back(t);
    return v;
  };

  // Initial per-residual-group b vector.
  auto b_outs = seg.rhs_loader->run(build_inputs(u));
  _assert(b_outs.size() == seg.residual_groups.size(),
          "rhs.pt2 must return ",
          seg.residual_groups.size(),
          " tensors (one per residual group), got ",
          b_outs.size());
  // Per-batch norm of the residual: sqrt of sum across residual groups
  // of sum over (group_sub + group_base) of group^2. Leading axes are
  // the dynamic-batch shape.
  auto b0_norm_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);
  auto b0_norm = b0_norm_sq.sqrt();

  const char * trace_env = std::getenv("NEML2_AOTI_TRACE_NEWTON");
  const int trace_level = trace_env ? std::atoi(trace_env) : 0;
  const bool trace = trace_level >= 1;
  const bool verbose = trace_level >= 2;

  if (check_converged(b0_norm, b0_norm, seg.atol, seg.rtol))
  {
    if (trace)
      std::cerr << "[aoti newton]iters=0 (converged at predictor) "
                << "b0_norm=" << b0_norm.max().item<double>() << std::endl;
    return u;
  }
  std::size_t reached = seg.miters;
  if (verbose)
    std::cerr << "ITERATION " << std::setw(3) << 0 << ", |R| = " << std::scientific
              << b0_norm.max().item<double>() << ", |R0| = " << std::scientific
              << b0_norm.max().item<double>() << std::endl;
  for (std::size_t i = 1; i < seg.miters; ++i)
  {
    // step graph returns (*du_groups, *b_curr_groups) of length n_u + n_r.
    // We already have b_outs from the prior iteration, so b_curr is
    // discarded (kept by the graph signature for symmetry / future use).
    const auto step_outs = seg.step_loader->run(build_inputs(u));
    _assert(step_outs.size() == seg.unknown_groups.size() + seg.residual_groups.size(),
            "step.pt2 must return n_unknown_groups + n_residual_groups tensors, got ",
            step_outs.size(),
            " (expected ",
            seg.unknown_groups.size() + seg.residual_groups.size(),
            ")");
    std::vector<at::Tensor> du(step_outs.begin(), step_outs.begin() + seg.unknown_groups.size());

    auto alpha = at::ones_like(b0_norm);
    // Armijo `b · du`: dot product across all residual+unknown groups.
    // For a square system, residual groups align with unknown groups by
    // position (residual_groups[k] is the residual block for
    // unknown_groups[k]), with matching per-group structure so trailing
    // axes match per pair.
    const auto b_dot_du = _pergroup_dot(b_outs, du, seg.unknown_groups);
    const auto nb_curr_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);

    std::vector<at::Tensor> u_trial(seg.unknown_groups.size());
    std::vector<at::Tensor> b_trial;
    if (seg.ls_max_iters <= 1)
    {
      for (std::size_t k = 0; k < seg.unknown_groups.size(); ++k)
        u_trial[k] = (u[k] + du[k]).contiguous();
      b_trial = seg.rhs_loader->run(build_inputs(u_trial));
      _assert(b_trial.size() == seg.residual_groups.size(),
              "rhs.pt2 must return ",
              seg.residual_groups.size(),
              " tensors (one per residual group)");
    }
    else
    {
      for (std::size_t k_ls = 1; k_ls < seg.ls_max_iters; ++k_ls)
      {
        for (std::size_t k = 0; k < seg.unknown_groups.size(); ++k)
        {
          const auto alpha_b = _alpha_for_group(alpha, seg.unknown_groups[k]);
          u_trial[k] = (u[k] + alpha_b * du[k]).contiguous();
        }
        b_trial = seg.rhs_loader->run(build_inputs(u_trial));
        _assert(b_trial.size() == seg.residual_groups.size(),
                "rhs.pt2 must return ",
                seg.residual_groups.size(),
                " tensors (one per residual group)");

        const auto nb_trial_sq = _pergroup_norm_sq(b_trial, seg.residual_groups);
        at::Tensor crit;
        if (seg.ls_type == "STRONG_WOLFE")
          crit = (1.0 - seg.ls_c * alpha) * nb_curr_sq;
        else // BACKTRACKING
          crit = nb_curr_sq - 2.0 * seg.ls_c * alpha * b_dot_du;

        if (verbose)
          std::cerr << "     LS ITERATION " << std::setw(3) << k_ls
                    << ", min(alpha) = " << std::scientific << alpha.min().item<double>()
                    << ", max(||R||) = " << std::scientific
                    << nb_trial_sq.sqrt().max().item<double>()
                    << ", min(||Rc||) = " << std::scientific << crit.sqrt().min().item<double>()
                    << std::endl;

        const auto stop = at::logical_or(nb_trial_sq <= crit, nb_trial_sq <= seg.atol * seg.atol);
        if (stop.all().item<bool>())
          break;
        alpha = at::where(stop, alpha, alpha / seg.ls_cutback);
      }
    }

    u = std::move(u_trial);
    b_outs = std::move(b_trial);

    const auto b_norm_sq = _pergroup_norm_sq(b_outs, seg.residual_groups);
    const auto b_norm = b_norm_sq.sqrt();
    if (verbose)
      std::cerr << "ITERATION " << std::setw(3) << i << ", |R| = " << std::scientific
                << b_norm.max().item<double>() << ", |R0| = " << std::scientific
                << b0_norm.max().item<double>() << std::endl;
    const auto status = check_stop(b_norm, b0_norm, seg.atol, seg.rtol);
    if (status == StopStatus::Diverged)
      throw std::runtime_error("AOTI Newton diverged at iter " + std::to_string(i) +
                               " (non-finite residual). Consider tightening the predictor, "
                               "increasing max_linesearch_iterations, or reducing the time step.");
    if (status == StopStatus::Converged)
    {
      reached = i;
      if (trace)
        std::cerr << "[aoti newton]iters=" << i << " (converged) "
                  << "b0_norm=" << b0_norm.max().item<double>()
                  << " b_norm=" << b_norm.max().item<double>() << std::endl;
      return u;
    }
  }

  if (trace)
  {
    const auto final_norm = _pergroup_norm_sq(b_outs, seg.residual_groups).sqrt();
    std::cerr << "[aoti newton]iters=" << reached
              << " (MAXITERS HIT) b0_norm=" << b0_norm.max().item<double>()
              << " b_norm=" << final_norm.max().item<double>() << std::endl;
  }

  // Maxed out without converging. Match the C++ Newton solver convention:
  // return the last iterate rather than throwing so callers can inspect it.
  return u;
}

} // namespace neml2::aoti
