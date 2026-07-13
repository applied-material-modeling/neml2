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
// Value path: forward-segment runner + the per-group implicit-segment driver
// ----------------------------------------------------------------------------
// The Newton iteration itself lives in newton.{h,cpp}; this file packs/unpacks
// per-group state and wires an AOTINonlinearSystem (compiled loaders) into the
// shared `Newton` solver.

#include "neml2/csrc/aoti/internal.h"
#include "neml2/csrc/aoti/newton.h"
#include "neml2/csrc/aoti/nonlinear_system_aoti.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
void
Model::Impl::_run_forward_segment(const Segment & seg,
                                  std::map<std::string, at::Tensor> & state,
                                  const std::vector<int64_t> & batch) const
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
  // Promoted-parameter tail: the forward value graph takes each parameter as a
  // per-batch input, so broadcast the stored parameter (scalar or already
  // batched) to the call batch before the call.
  for (const auto & pname : seg.param_inputs)
    inputs.push_back(broadcast_param_to_batch(
        _resolve_param(pname), batch, static_cast<int64_t>(_param_base_shapes.at(pname).size())));

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

  // Seed each unknown's initial guess from its incoming state value -- the
  // caller's warm start for a standalone segment, the upstream value in a
  // composed graph -- matching the eager route, whose
  // `ImplicitUpdate._initial_unknowns` returns `state[name]` when there is no
  // predictor. Only fall back to zeros when the unknown has no incoming value;
  // seeding zeros unconditionally (the previous behavior) diverged the Newton on
  // stiff no-predictor models that eager solves fine (a parity violation). A
  // predictor, if present, overrides below.
  for (const auto & v : seg.unknowns)
    if (state.find(v.name) == state.end())
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
    // The predictor is compiled without the residual's promoted tail; pass its
    // own (currently always empty) promoted-param list, not seg.param_inputs.
    for (auto & p : _gather_params(seg.predictor_param_inputs))
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

  // Drive the shared Newton solver over an AOTI-backed system. The givens +
  // promoted-parameter tail are bound into the system (constant across the
  // solve); the solver config comes from the segment metadata.
  auto to_layouts = [](const std::vector<Segment::GroupInfo> & groups)
  {
    std::vector<GroupLayout> out;
    out.reserve(groups.size());
    for (const auto & g : groups)
      out.push_back(GroupLayout{g.structure, g.sub_batch_shape});
    return out;
  };
  AOTINonlinearSystem sys(*seg.rhs_loader,
                          *seg.step_loader,
                          to_layouts(seg.unknown_groups),
                          to_layouts(seg.residual_groups),
                          g_groups,
                          _gather_params(seg.param_inputs));
  // Solver config is read from the shared metadata.json at construction
  // (overridable via set_solver_config). A failed solve (divergence or max-iterations) throws
  // ConvergenceError out of solve(), so reaching `.u` means it converged -- the
  // recoverable error propagates up through Model::forward/jacobian to the
  // caller, who can cut the time step and retry.
  u_solved_groups = Newton(_solver_config).solve(sys, u0_groups).u;

  // Unpack converged per-group unknowns back to per-variable state for
  // downstream forward segments / master outputs to read by name.
  _unpack_groups(u_solved_groups, seg.unknown_groups, state);
}

at::Tensor
Model::Impl::_run_implicit_segment_masked(const Segment & seg,
                                          std::map<std::string, at::Tensor> & state) const
{
  // Same seed + predictor + pack path as `_run_implicit_segment`, but drives
  // `Newton::solve_masked` (returns the per-element convergence mask, no throw)
  // so the substep driver can freeze converged rows and bisect the rest.
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

  // Warm-start each unknown from its incoming state value (parity with eager's
  // `ImplicitUpdate._initial_unknowns`); zeros only when none is present. See the
  // value-path seed above for the rationale.
  for (const auto & v : seg.unknowns)
    if (state.find(v.name) == state.end())
      state[v.name] = at::zeros(_full_shape(v), opts);

  if (seg.predictor_loader)
  {
    std::vector<at::Tensor> p_inputs;
    p_inputs.reserve(seg.predictor_inputs.size() + seg.predictor_param_inputs.size());
    for (const auto & name : seg.predictor_inputs)
    {
      auto it = state.find(name);
      _assert(it != state.end(),
              "aoti::Model: implicit segment predictor needs input '",
              name,
              "' which is not in the state map.");
      p_inputs.push_back(it->second.contiguous());
    }
    for (auto & p : _gather_params(seg.predictor_param_inputs))
      p_inputs.push_back(std::move(p));
    const auto p_outs = seg.predictor_loader->run(p_inputs);
    _assert(p_outs.size() == seg.predictor_outputs.size(),
            "aoti::Model: predictor returned ",
            p_outs.size(),
            " outputs, expected ",
            seg.predictor_outputs.size());
    for (std::size_t i = 0; i < p_outs.size(); ++i)
      state[seg.predictor_outputs[i]] = p_outs[i].to(opts).contiguous();
  }

  auto g_groups = _pack_groups(state, seg.given_groups);
  auto u0_groups = _pack_groups(state, seg.unknown_groups);

  auto to_layouts = [](const std::vector<Segment::GroupInfo> & groups)
  {
    std::vector<GroupLayout> out;
    out.reserve(groups.size());
    for (const auto & g : groups)
      out.push_back(GroupLayout{g.structure, g.sub_batch_shape});
    return out;
  };
  AOTINonlinearSystem sys(*seg.rhs_loader,
                          *seg.step_loader,
                          to_layouts(seg.unknown_groups),
                          to_layouts(seg.residual_groups),
                          g_groups,
                          _gather_params(seg.param_inputs));
  auto res = Newton(_solver_config).solve_masked(sys, u0_groups);
  _unpack_groups(res.u, seg.unknown_groups, state);
  return res.converged_mask;
}
} // namespace neml2::aoti
