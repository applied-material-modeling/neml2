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
// Adaptive substepping driver (host-side control flow over the compiled solve)
// ----------------------------------------------------------------------------
// The single sub-step solve / IFT stays the compiled unit (`_run_implicit_segment`
// / `_run_implicit_segment_jacobian`); the bisection-on-failure, force
// interpolation, increment scaling, old-state chaining, and consistent-tangent
// accumulation are host control flow here -- exactly how the Newton iteration
// itself is orchestrated. Data-dependent (which sub-steps fail) control flow
// cannot live in a compiled graph, so it lives here. See substepping_epic.

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/internal.h"
#include "neml2/csrc/aoti/log.h"
#include "neml2/csrc/dispatchers/batch_chunk.h"

#include <functional>
#include <sstream>

// Alias so the logging namespace is reachable as ``nlog``.
namespace nlog = neml2::aoti::log;

namespace neml2::aoti
{
void
Model::Impl::_apply_substep_span(const Segment & seg,
                                 const std::map<std::string, at::Tensor> & orig,
                                 double a,
                                 double b,
                                 const std::map<std::string, at::Tensor> & chained,
                                 std::map<std::string, at::Tensor> & dst) const
{
  // Paired forces interpolate linearly between the increment's endpoints (so the
  // deformation *rate* over dt/M is unchanged -- time is itself an old/cur force
  // pair, so dt subdivides automatically); an old-state takes the chained value;
  // anything else is held at its endpoint value. The same coefficients apply to
  // the primal state and to the dstate (chain-rule) carrier -- that is why this
  // is shared.
  for (const auto & g : seg.givens)
  {
    if (g.role == "old_state")
    {
      auto it = chained.find(g.name);
      _assert(it != chained.end(),
              "aoti::Model substepping: no chained value for old-state given '",
              g.name,
              "'.");
      dst[g.name] = it->second;
    }
    else if (g.role == "old_force")
    {
      // name is the OLD endpoint (~1); pair (the current force) is the NEW.
      const auto & old_e = orig.at(g.name);
      const auto & new_e = orig.at(g.pair);
      dst[g.name] = old_e + (new_e - old_e) * a;
    }
    else if (g.role == "cur_force")
    {
      // pair (the ~1 force) is the OLD endpoint; name is the NEW.
      const auto & old_e = orig.at(g.pair);
      const auto & new_e = orig.at(g.name);
      dst[g.name] = old_e + (new_e - old_e) * b;
    }
    else
    {
      // static / unknown / unrecognized: hold the endpoint value.
      dst[g.name] = orig.at(g.name);
    }
  }
}

void
Model::Impl::_run_implicit_segment_substepped(const Segment & seg,
                                              std::map<std::string, at::Tensor> & state,
                                              std::vector<at::Tensor> & u_solved_groups,
                                              std::vector<at::Tensor> & g_groups) const
{
  // Snapshot the endpoint value of every given before we start overwriting them.
  std::map<std::string, at::Tensor> orig;
  for (const auto & g : seg.givens)
  {
    auto it = state.find(g.name);
    _assert(it != state.end(),
            "aoti::Model substepping: implicit segment given '",
            g.name,
            "' is not in the state map.");
    orig[g.name] = it->second;
  }

  std::function<void(double, double, const std::map<std::string, at::Tensor> &, int)> solve_span =
      [&](double a, double b, const std::map<std::string, at::Tensor> & chained_old, int level)
  {
    _apply_substep_span(seg, orig, a, b, chained_old, state);
    try
    {
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      return;
    }
    catch (const ConvergenceError &)
    {
      // Only a recoverable divergence / max-iters is retried by subdivision; a
      // FatalError (shape/device/config) propagates immediately.
      if (level >= seg.max_substepping_level)
        throw;
    }
    const double mid = 0.5 * (a + b);
    solve_span(a, mid, chained_old, level + 1);
    // The first half's converged unknowns now sit in `state`; feed each as the
    // second half's old-state (chain old-state given -> its base unknown).
    std::map<std::string, at::Tensor> chained_next;
    for (const auto & g : seg.givens)
      if (g.role == "old_state")
        chained_next[g.name] = state.at(g.pair);
    solve_span(mid, b, chained_next, level + 1);
  };

  // Initial chained old-state = the increment's own ~1 values (the true history).
  std::map<std::string, at::Tensor> chained0;
  for (const auto & g : seg.givens)
    if (g.role == "old_state")
      chained0[g.name] = orig.at(g.name);

  solve_span(0.0, 1.0, chained0, 0);
}

void
Model::Impl::_run_implicit_segment_substepped_jacobian(
    const Segment & seg,
    std::map<std::string, at::Tensor> & state,
    std::map<std::string, at::Tensor> & dstate) const
{
  // Snapshot endpoint VALUES (for the primal interpolation) and endpoint DSTATE
  // (the givens' sensitivity to the master inputs, for the tangent interpolation)
  // before either is overwritten per sub-span.
  std::map<std::string, at::Tensor> orig_val;
  std::map<std::string, at::Tensor> orig_dstate;
  for (const auto & g : seg.givens)
  {
    auto vit = state.find(g.name);
    _assert(vit != state.end(),
            "aoti::Model substepping: implicit segment given '",
            g.name,
            "' is not in the state map.");
    orig_val[g.name] = vit->second;
    auto dit = dstate.find(g.name);
    _assert(dit != dstate.end(),
            "aoti::Model substepping: dstate missing given '",
            g.name,
            "' at Jacobian composition time.");
    orig_dstate[g.name] = dit->second;
  }

  std::function<void(double,
                     double,
                     const std::map<std::string, at::Tensor> &,
                     const std::map<std::string, at::Tensor> &,
                     int)>
      solve_span = [&](double a,
                       double b,
                       const std::map<std::string, at::Tensor> & chained_val,
                       const std::map<std::string, at::Tensor> & chained_dstate,
                       int level)
  {
    _apply_substep_span(seg, orig_val, a, b, chained_val, state);
    _apply_substep_span(seg, orig_dstate, a, b, chained_dstate, dstate);
    try
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      // Overwrites dstate[unknown] with Σ_g (-A⁻¹B)_{u,g} · dstate[g]. With
      // dstate[old_state] carrying J_{k-1} and dstate[forces] the interpolated
      // endpoint sensitivities, this is exactly A_k·J_{k-1} + B_k·frac_k·J_endpoint.
      _run_implicit_segment_jacobian(seg, u_solved_groups, g_groups, dstate);
      return;
    }
    catch (const ConvergenceError &)
    {
      if (level >= seg.max_substepping_level)
        throw;
    }
    const double mid = 0.5 * (a + b);
    solve_span(a, mid, chained_val, chained_dstate, level + 1);
    // Chain BOTH the solved values and their sensitivities into the next half.
    std::map<std::string, at::Tensor> chained_val_next;
    std::map<std::string, at::Tensor> chained_dstate_next;
    for (const auto & g : seg.givens)
      if (g.role == "old_state")
      {
        chained_val_next[g.name] = state.at(g.pair);
        chained_dstate_next[g.name] = dstate.at(g.pair);
      }
    solve_span(mid, b, chained_val_next, chained_dstate_next, level + 1);
  };

  std::map<std::string, at::Tensor> chained0_val;
  std::map<std::string, at::Tensor> chained0_dstate;
  for (const auto & g : seg.givens)
    if (g.role == "old_state")
    {
      chained0_val[g.name] = orig_val.at(g.name);
      chained0_dstate[g.name] = orig_dstate.at(g.name);
    }

  solve_span(0.0, 1.0, chained0_val, chained0_dstate, 0);
}

// ----------------------------------------------------------------------------
// Per-element (masked) substepping
// ----------------------------------------------------------------------------
// Solve only the still-unconverged subset of the dynamic batch at each sub-step,
// freeze converged rows at their coarsest converging solution, and bisect only
// the failing subset -- so a few hard elements no longer drag the whole batch
// through the deep sub-steps. `a,b` stay scalar per span (uniform interval);
// masking is purely *which rows are active*, so `_apply_substep_span` and the
// coefficient math are reused unchanged. Requires a 1-D dynamic batch.

namespace
{
// Indices of the true rows of a 1-D bool mask (empty if none).
inline at::Tensor
mask_to_idx(const at::Tensor & m)
{
  return at::nonzero(m).squeeze(-1);
}

} // namespace

bool
Model::Impl::_masking_ok(const Segment & seg, const std::map<std::string, at::Tensor> & state) const
{
  if (seg.givens.empty())
    return false;
  auto it = state.find(seg.givens[0].name);
  if (it == state.end())
    return false;
  const auto & g0 = it->second;
  const auto & g0i = seg.givens[0];
  const int64_t trail = static_cast<int64_t>(g0i.sub_batch_shape.size() + g0i.base_shape.size());
  // Masking indexes dim 0; it is well-defined only when the dynamic batch is a
  // single leading axis (the batch_chunk.h contract). Multi-axis dynamic batches
  // fall back to the whole-batch driver.
  return (g0.dim() - trail) == 1;
}

void
Model::Impl::_run_implicit_segment_substepped_masked(
    const Segment & seg, std::map<std::string, at::Tensor> & state) const
{
  std::map<std::string, at::Tensor> orig;
  for (const auto & g : seg.givens)
    orig[g.name] = state.at(g.name);

  const auto & g0 = orig.at(seg.givens[0].name);
  const auto & g0i = seg.givens[0];
  const int64_t g0_trail = static_cast<int64_t>(g0i.sub_batch_shape.size() + g0i.base_shape.size());
  std::vector<int64_t> batch_shape(g0.sizes().begin(), g0.sizes().end() - g0_trail);
  const int64_t B = batch_shape[0];
  const auto opts = g0.options();
  auto full_shape = [&](const Segment::VarInfo & v)
  {
    std::vector<int64_t> s(batch_shape);
    s.insert(s.end(), v.sub_batch_shape.begin(), v.sub_batch_shape.end());
    s.insert(s.end(), v.base_shape.begin(), v.base_shape.end());
    return s;
  };

  // Full-batch running buffers: chained old-state values + solved unknowns.
  std::map<std::string, at::Tensor> chained;
  for (const auto & g : seg.givens)
    if (g.role == "old_state")
      chained[g.name] = orig.at(g.name).clone();
  std::map<std::string, at::Tensor> result;
  for (const auto & u : seg.unknowns)
    result[u.name] = at::zeros(full_shape(u), opts);

  const auto idx_opts = at::TensorOptions().dtype(at::kLong).device(g0.device());

  const bool console_info = nlog::enabled(nlog::Channel::Substep, nlog::Level::Info);
  const bool console_debug = nlog::enabled(nlog::Channel::Substep, nlog::Level::Debug);
  int64_t n_solves = 0, max_depth = 0, n_substepped = 0;
  std::function<void(double, double, const at::Tensor &, int)> solve_to =
      [&](double a, double b, const at::Tensor & active, int level)
  {
    auto orig_a = index_select_batch(orig, active);
    auto chained_a = index_select_batch(chained, active);
    std::map<std::string, at::Tensor> span;
    _apply_substep_span(seg, orig_a, a, b, chained_a, span);
    auto mask = _run_implicit_segment_masked(seg, span);
    auto conv = mask_to_idx(mask);
    auto fail = mask_to_idx(at::logical_not(mask));
    ++n_solves;
    max_depth = std::max(max_depth, static_cast<int64_t>(level));
    if (level == 0)
      n_substepped = fail.numel(); // rows that failed the full step need substepping
    // LCOV_EXCL_START -- diagnostic per-sub-span trace (verbosity-gated; see neml2.log)
    if (console_debug)
    {
      std::ostringstream oss;
      // Indent by bisection depth so the sub-span tree is visible at a glance.
      oss << std::string(static_cast<std::size_t>(2 * (level + 1)), ' ') << "span [" << a << ", "
          << b << "] L" << level << ": active=" << active.numel()
          << " -> converged=" << conv.numel() << " failed=" << fail.numel();
      nlog::emit(nlog::Channel::Substep, nlog::Level::Debug, oss.str());
    }
    // LCOV_EXCL_STOP
    if (conv.numel() > 0)
    {
      auto conv_g = active.index_select(0, conv);
      std::map<std::string, at::Tensor> cs, cc;
      for (const auto & u : seg.unknowns)
        cs[u.name] = span.at(u.name).index_select(0, conv);
      scatter_batch_(result, conv_g, cs);
      for (const auto & g : seg.givens)
        if (g.role == "old_state")
          cc[g.name] = span.at(g.pair).index_select(0, conv);
      scatter_batch_(chained, conv_g, cc);
    }
    if (fail.numel() > 0)
    {
      // Maxing out substepping is a RECOVERABLE convergence failure -- a
      // time-stepping consumer (e.g. MOOSE) cuts the outer step and retries -- so
      // it must throw the recoverable ConvergenceError, NOT `_assert` (which
      // throws the non-recoverable FatalError, defeating a `recoverable()` retry
      // and making a maxed-out substep unrecoverable downstream). This mirrors the
      // whole-batch drivers, which re-throw the caught ConvergenceError.
      if (level >= seg.max_substepping_level)
        throw ConvergenceError("aoti::Model substepping: " + std::to_string(fail.numel()) +
                               " element(s) failed to converge at max_substepping_level=" +
                               std::to_string(seg.max_substepping_level) +
                               ". Reduce the outer time step.");
      auto fail_g = active.index_select(0, fail);
      const double mid = 0.5 * (a + b);
      solve_to(a, mid, fail_g, level + 1);
      solve_to(mid, b, fail_g, level + 1);
    }
  };

  solve_to(0.0, 1.0, at::arange(B, idx_opts), 0);
  // LCOV_EXCL_START -- diagnostic per-solve substep summary
  if (console_info)
  {
    std::ostringstream oss;
    oss << "value: B=" << B << " elements, " << n_substepped
        << " substepped, max depth=" << max_depth << ", " << n_solves << " segment-solves";
    nlog::emit(nlog::Channel::Substep, nlog::Level::Info, oss.str());
  }
  // LCOV_EXCL_STOP
  for (const auto & u : seg.unknowns)
    state[u.name] = result[u.name];
}

void
Model::Impl::_run_implicit_segment_substepped_masked_jacobian(
    const Segment & seg,
    std::map<std::string, at::Tensor> & state,
    std::map<std::string, at::Tensor> & dstate) const
{
  std::map<std::string, at::Tensor> orig, orig_d;
  for (const auto & g : seg.givens)
  {
    orig[g.name] = state.at(g.name);
    orig_d[g.name] = dstate.at(g.name);
  }

  const auto & g0 = orig.at(seg.givens[0].name);
  const auto & g0i = seg.givens[0];
  const int64_t g0_trail = static_cast<int64_t>(g0i.sub_batch_shape.size() + g0i.base_shape.size());
  std::vector<int64_t> batch_shape(g0.sizes().begin(), g0.sizes().end() - g0_trail);
  const int64_t B = batch_shape[0];
  const auto opts = g0.options();
  const int64_t M = orig_d.at(seg.givens[0].name).size(-1);
  auto full_shape = [&](const Segment::VarInfo & v)
  {
    std::vector<int64_t> s(batch_shape);
    s.insert(s.end(), v.sub_batch_shape.begin(), v.sub_batch_shape.end());
    s.insert(s.end(), v.base_shape.begin(), v.base_shape.end());
    return s;
  };

  std::map<std::string, at::Tensor> chained, chained_d;
  for (const auto & g : seg.givens)
    if (g.role == "old_state")
    {
      chained[g.name] = orig.at(g.name).clone();
      chained_d[g.name] = orig_d.at(g.name).clone();
    }
  std::map<std::string, at::Tensor> result, result_d;
  for (const auto & u : seg.unknowns)
  {
    result[u.name] = at::zeros(full_shape(u), opts);
    std::vector<int64_t> sd(batch_shape);
    sd.push_back(u.var_size);
    sd.push_back(M);
    result_d[u.name] = at::zeros(sd, opts);
  }

  const auto idx_opts = at::TensorOptions().dtype(at::kLong).device(g0.device());

  const bool console_info = nlog::enabled(nlog::Channel::Substep, nlog::Level::Info);
  const bool console_debug = nlog::enabled(nlog::Channel::Substep, nlog::Level::Debug);
  int64_t n_solves = 0, max_depth = 0, n_substepped = 0;
  std::function<void(double, double, const at::Tensor &, int)> solve_to =
      [&](double a, double b, const at::Tensor & active, int level)
  {
    auto orig_a = index_select_batch(orig, active);
    auto chained_a = index_select_batch(chained, active);
    auto orig_da = index_select_batch(orig_d, active);
    auto chained_da = index_select_batch(chained_d, active);
    std::map<std::string, at::Tensor> span, span_d;
    _apply_substep_span(seg, orig_a, a, b, chained_a, span);
    _apply_substep_span(seg, orig_da, a, b, chained_da, span_d);
    auto mask = _run_implicit_segment_masked(seg, span);
    auto conv = mask_to_idx(mask);
    auto fail = mask_to_idx(at::logical_not(mask));
    ++n_solves;
    max_depth = std::max(max_depth, static_cast<int64_t>(level));
    if (level == 0)
      n_substepped = fail.numel();
    // LCOV_EXCL_START -- diagnostic per-sub-span trace (verbosity-gated; see neml2.log)
    if (console_debug)
    {
      std::ostringstream oss;
      // Indent by bisection depth so the sub-span tree is visible at a glance.
      oss << std::string(static_cast<std::size_t>(2 * (level + 1)), ' ') << "span [" << a << ", "
          << b << "] L" << level << ": active=" << active.numel()
          << " -> converged=" << conv.numel() << " failed=" << fail.numel();
      nlog::emit(nlog::Channel::Substep, nlog::Level::Debug, oss.str());
    }
    // LCOV_EXCL_STOP
    if (conv.numel() > 0)
    {
      auto conv_g = active.index_select(0, conv);
      auto conv_span = index_select_batch(span, conv);
      auto conv_span_d = index_select_batch(span_d, conv);
      auto u_groups = _pack_groups(conv_span, seg.unknown_groups);
      auto g_groups = _pack_groups(conv_span, seg.given_groups);
      // Overwrites conv_span_d[unknown] = Σ (-A⁻¹B)_{u,g}·conv_span_d[g]
      //   = A_k·J_{k-1} + B_k·frac_k·J_endpoint for this span's converged rows.
      _run_implicit_segment_jacobian(seg, u_groups, g_groups, conv_span_d);
      std::map<std::string, at::Tensor> cs, csd, cc, ccd;
      for (const auto & u : seg.unknowns)
      {
        cs[u.name] = conv_span.at(u.name);
        csd[u.name] = conv_span_d.at(u.name);
      }
      scatter_batch_(result, conv_g, cs);
      scatter_batch_(result_d, conv_g, csd);
      for (const auto & g : seg.givens)
        if (g.role == "old_state")
        {
          cc[g.name] = conv_span.at(g.pair);
          ccd[g.name] = conv_span_d.at(g.pair);
        }
      scatter_batch_(chained, conv_g, cc);
      scatter_batch_(chained_d, conv_g, ccd);
    }
    if (fail.numel() > 0)
    {
      // Maxing out substepping is a RECOVERABLE convergence failure -- a
      // time-stepping consumer (e.g. MOOSE) cuts the outer step and retries -- so
      // it must throw the recoverable ConvergenceError, NOT `_assert` (which
      // throws the non-recoverable FatalError, defeating a `recoverable()` retry
      // and making a maxed-out substep unrecoverable downstream). This mirrors the
      // whole-batch drivers, which re-throw the caught ConvergenceError.
      if (level >= seg.max_substepping_level)
        throw ConvergenceError("aoti::Model substepping: " + std::to_string(fail.numel()) +
                               " element(s) failed to converge at max_substepping_level=" +
                               std::to_string(seg.max_substepping_level) +
                               ". Reduce the outer time step.");
      auto fail_g = active.index_select(0, fail);
      const double mid = 0.5 * (a + b);
      solve_to(a, mid, fail_g, level + 1);
      solve_to(mid, b, fail_g, level + 1);
    }
  };

  solve_to(0.0, 1.0, at::arange(B, idx_opts), 0);
  // LCOV_EXCL_START -- diagnostic per-solve substep summary
  if (console_info)
  {
    std::ostringstream oss;
    oss << "jacobian: B=" << B << " elements, " << n_substepped
        << " substepped, max depth=" << max_depth << ", " << n_solves << " segment-solves";
    nlog::emit(nlog::Channel::Substep, nlog::Level::Info, oss.str());
  }
  // LCOV_EXCL_STOP
  for (const auto & u : seg.unknowns)
  {
    state[u.name] = result[u.name];
    dstate[u.name] = result_d[u.name];
  }
}
} // namespace neml2::aoti
