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

#include <functional>

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
  // pair, so dt subdivides automatically); a listed increment is scaled by the
  // span fraction (b - a); an old-state takes the chained value; anything else is
  // held at its endpoint value. The same coefficients apply to the primal state
  // and to the dstate (chain-rule) carrier -- that is why this is shared.
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
    else if (g.role == "incremental")
    {
      dst[g.name] = orig.at(g.name) * (b - a);
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
} // namespace neml2::aoti
