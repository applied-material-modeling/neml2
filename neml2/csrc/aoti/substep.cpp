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
// The single sub-step solve stays the compiled unit (`_run_implicit_segment`);
// the bisection-on-failure, force interpolation, increment scaling, and
// old-state chaining are host control flow here -- exactly how the Newton
// iteration itself is orchestrated. Data-dependent (which sub-steps fail)
// control flow cannot live in a compiled graph, so it lives here.

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/internal.h"

#include <functional>

namespace neml2::aoti
{
void
Model::Impl::_run_implicit_segment_substepped(const Segment & seg,
                                              std::map<std::string, at::Tensor> & state,
                                              std::vector<at::Tensor> & u_solved_groups,
                                              std::vector<at::Tensor> & g_groups) const
{
  // Snapshot the endpoint value of every given before we start overwriting them
  // per sub-span; interpolation / scaling reads these originals.
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

  // Write the givens for the sub-span [a, b] into `state`, per each given's role.
  // Paired forces interpolate linearly between the increment's endpoints (so the
  // deformation *rate* over dt/M is unchanged -- time is itself an old/cur force
  // pair, subdividing dt automatically); a listed increment is scaled by the
  // span fraction (b - a); an old-state is the chained value; anything else is
  // held at its endpoint value.
  auto set_span_givens =
      [&](double a, double b, const std::map<std::string, at::Tensor> & chained_old)
  {
    for (const auto & g : seg.givens)
    {
      if (g.role == "old_state")
      {
        auto it = chained_old.find(g.name);
        _assert(it != chained_old.end(),
                "aoti::Model substepping: no chained value for old-state given '",
                g.name,
                "'.");
        state[g.name] = it->second;
      }
      else if (g.role == "old_force")
      {
        // name is the OLD endpoint (~1); pair (the current force) is the NEW.
        const auto & old_e = orig.at(g.name);
        const auto & new_e = orig.at(g.pair);
        state[g.name] = old_e + (new_e - old_e) * a;
      }
      else if (g.role == "cur_force")
      {
        // pair (the ~1 force) is the OLD endpoint; name is the NEW.
        const auto & old_e = orig.at(g.pair);
        const auto & new_e = orig.at(g.name);
        state[g.name] = old_e + (new_e - old_e) * b;
      }
      else if (g.role == "incremental")
      {
        state[g.name] = orig.at(g.name) * (b - a);
      }
      else
      {
        // static / unknown / unrecognized: hold the endpoint value.
        state[g.name] = orig.at(g.name);
      }
    }
  };

  // Recursively solve the span [a, b]; on a recoverable failure bisect (up to
  // `max_substepping_level` deep), chaining the solved state across the halves.
  std::function<void(double, double, const std::map<std::string, at::Tensor> &, int)> solve_span =
      [&](double a, double b, const std::map<std::string, at::Tensor> & chained_old, int level)
  {
    set_span_givens(a, b, chained_old);
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
} // namespace neml2::aoti
