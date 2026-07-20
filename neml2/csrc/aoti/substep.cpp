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

// ----------------------------------------------------------------------------
// Per-element (masked) substepping -- the ONLY substepping path
// ----------------------------------------------------------------------------
// Solve only the still-unconverged subset of the dynamic batch at each sub-step,
// freeze converged rows at their coarsest converging solution, and bisect only
// the failing subset -- so a few hard elements no longer drag the whole batch
// through the deep sub-steps. `a,b` stay scalar per span (uniform interval);
// masking is purely *which rows are active*, so `_apply_substep_span` and the
// coefficient math are reused unchanged. Masking indexes dim 0, so the drivers
// flatten the dynamic batch to a single leading axis (any rank -> 1-D; an
// unbatched call -> a leading 1) before running and reshape the solved unknowns
// back on return -- the compiled solve accepts any dynamic-batch rank, so this
// is a pure reshape (see `flatten_dyn` / `unflatten_dyn`).

namespace
{
// Indices of the true rows of a 1-D bool mask (empty if none).
inline at::Tensor
mask_to_idx(const at::Tensor & m)
{
  return at::nonzero(m).squeeze(-1);
}

// Trailing (sub_batch + base) axis count of a segment given/unknown -- the axes
// that sit behind the dynamic batch. Works for both `seg.givens` and
// `seg.unknowns` (both carry `sub_batch_shape` + `base_shape`).
template <typename VarT>
int64_t
var_trail(const VarT & v)
{
  return static_cast<int64_t>(v.sub_batch_shape.size() + v.base_shape.size());
}

// Flatten the leading dynamic-batch axes of `t` (everything before its `trail`
// trailing axes) into a single axis, so the masking driver's dim-0
// index/scatter is well-defined for any dynamic-batch rank (0-D -> a leading 1).
inline at::Tensor
flatten_dyn(const at::Tensor & t, int64_t trail)
{
  std::vector<int64_t> shape;
  shape.reserve(static_cast<std::size_t>(trail) + 1);
  shape.push_back(-1);
  for (int64_t d = t.dim() - trail; d < t.dim(); ++d)
    shape.push_back(t.size(d));
  return t.reshape(shape);
}

// Inverse of `flatten_dyn`: restore the dynamic batch `dyn` in front of the last
// `trail` trailing axes of `t` (whose leading axis is the flattened batch).
inline at::Tensor
unflatten_dyn(const at::Tensor & t, const std::vector<int64_t> & dyn, int64_t trail)
{
  std::vector<int64_t> shape(dyn);
  for (int64_t d = t.dim() - trail; d < t.dim(); ++d)
    shape.push_back(t.size(d));
  return t.reshape(shape);
}

} // namespace

void
Model::Impl::_run_implicit_segment_substepped_masked(
    const Segment & seg, std::map<std::string, at::Tensor> & state) const
{
  const bool capture = capture_solve_failure_enabled();

  // Flatten the dynamic batch to a single leading axis (see the file-level note);
  // restored / reshaped back on return. `dyn` is the original dynamic batch.
  const int64_t g0_dyn_trail = var_trail(seg.givens[0]);
  const auto & g0_orig = state.at(seg.givens[0].name);
  const std::vector<int64_t> dyn(g0_orig.sizes().begin(), g0_orig.sizes().end() - g0_dyn_trail);
  const bool reshape_needed = dyn.size() != 1;
  std::map<std::string, at::Tensor> saved_givens;
  if (reshape_needed)
    for (const auto & g : seg.givens)
    {
      saved_givens[g.name] = state.at(g.name);
      state[g.name] = flatten_dyn(saved_givens[g.name], var_trail(g));
    }

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
  // Opt-in failure context (attached to the recoverable error if substepping
  // ultimately fails): the level-0 single-shot mask + best-effort iterate.
  at::Tensor cap_mask;
  std::map<std::string, at::Tensor> cap_unknowns;
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
    if (capture && level == 0)
    {
      // Level 0 = the single-shot masked solve over the full increment (all rows
      // active): its mask + best-effort iterate are the failure context, matching
      // the non-substepped capture in solve.cpp.
      cap_mask = mask;
      for (const auto & u : seg.unknowns)
        cap_unknowns[u.name] = span.at(u.name);
    }
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
      // and making a maxed-out substep unrecoverable downstream).
      if (level >= seg.max_substepping_level)
      {
        const std::string msg = "aoti::Model substepping: " + std::to_string(fail.numel()) +
                                " element(s) failed to converge at max_substepping_level=" +
                                std::to_string(seg.max_substepping_level) +
                                ". Reduce the outer time step.";
        // Attach the level-0 (single-shot) failure context when capture is on,
        // reshaped back to the original dynamic batch.
        if (capture)
        {
          std::map<std::string, at::Tensor> stuck;
          for (const auto & u : seg.unknowns)
            stuck[u.name] = reshape_needed
                                ? unflatten_dyn(cap_unknowns.at(u.name), dyn, var_trail(u))
                                : cap_unknowns.at(u.name);
          throw ConvergenceError(
              msg, reshape_needed ? cap_mask.reshape(dyn) : cap_mask, std::move(stuck));
        }
        throw ConvergenceError(msg);
      }
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

  // Restore original-shape givens + reshape the solved unknowns back to the call
  // batch (the driver ran on the flattened 1-D dynamic batch).
  if (reshape_needed)
  {
    for (const auto & g : seg.givens)
      state[g.name] = saved_givens[g.name];
    for (const auto & u : seg.unknowns)
      state[u.name] = unflatten_dyn(state.at(u.name), dyn, var_trail(u));
  }
}

void
Model::Impl::_run_implicit_segment_substepped_masked_jacobian(
    const Segment & seg,
    std::map<std::string, at::Tensor> & state,
    std::map<std::string, at::Tensor> & dstate) const
{
  const bool capture = capture_solve_failure_enabled();

  // Flatten the dynamic batch to a single leading axis (see the file-level note).
  // A dstate block is (*dyn, folded, M) -- the pure dynamic batch (sub-batch
  // stripped; see `_init_dstate`) plus exactly two trailing axes (folded storage,
  // derivative width M) -- so its trail is 2, regardless of the primal given's.
  const int64_t g0_dyn_trail = var_trail(seg.givens[0]);
  const auto & g0_orig = state.at(seg.givens[0].name);
  const std::vector<int64_t> dyn(g0_orig.sizes().begin(), g0_orig.sizes().end() - g0_dyn_trail);
  const bool reshape_needed = dyn.size() != 1;
  std::map<std::string, at::Tensor> saved_givens, saved_givens_d;
  if (reshape_needed)
    for (const auto & g : seg.givens)
    {
      saved_givens[g.name] = state.at(g.name);
      saved_givens_d[g.name] = dstate.at(g.name);
      state[g.name] = flatten_dyn(saved_givens[g.name], var_trail(g));
      dstate[g.name] = flatten_dyn(saved_givens_d[g.name], /*trail=*/2);
    }

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
  // Opt-in failure context (level-0 single-shot mask + best-effort value iterate).
  at::Tensor cap_mask;
  std::map<std::string, at::Tensor> cap_unknowns;
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
    if (capture && level == 0)
    {
      // Level 0 = the single-shot masked solve; its mask + best-effort value
      // iterate are the failure context (same semantics as solve.cpp).
      cap_mask = mask;
      for (const auto & u : seg.unknowns)
        cap_unknowns[u.name] = span.at(u.name);
    }
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
      // and making a maxed-out substep unrecoverable downstream).
      if (level >= seg.max_substepping_level)
      {
        const std::string msg = "aoti::Model substepping: " + std::to_string(fail.numel()) +
                                " element(s) failed to converge at max_substepping_level=" +
                                std::to_string(seg.max_substepping_level) +
                                ". Reduce the outer time step.";
        // Attach the level-0 (single-shot) failure context when capture is on,
        // reshaped back to the original dynamic batch.
        if (capture)
        {
          std::map<std::string, at::Tensor> stuck;
          for (const auto & u : seg.unknowns)
            stuck[u.name] = reshape_needed
                                ? unflatten_dyn(cap_unknowns.at(u.name), dyn, var_trail(u))
                                : cap_unknowns.at(u.name);
          throw ConvergenceError(
              msg, reshape_needed ? cap_mask.reshape(dyn) : cap_mask, std::move(stuck));
        }
        throw ConvergenceError(msg);
      }
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

  // Restore original-shape givens (value + dstate) and reshape the solved
  // unknowns + their tangents (*, var_size, M) back to the call batch.
  if (reshape_needed)
  {
    for (const auto & g : seg.givens)
    {
      state[g.name] = saved_givens[g.name];
      dstate[g.name] = saved_givens_d[g.name];
    }
    for (const auto & u : seg.unknowns)
    {
      state[u.name] = unflatten_dyn(state.at(u.name), dyn, var_trail(u));
      dstate[u.name] = unflatten_dyn(dstate.at(u.name), dyn, /*trail=*/2);
    }
  }
}
} // namespace neml2::aoti
