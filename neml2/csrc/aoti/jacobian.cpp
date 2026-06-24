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
// Jacobian path: dstate seeding + forward-segment JVP composition + implicit
// IFT composition
// ----------------------------------------------------------------------------

#include "neml2/csrc/aoti/internal.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
void
Model::Impl::_init_dstate(const at::TensorOptions & options,
                          at::IntArrayRef batch_shape,
                          std::map<std::string, at::Tensor> & dstate) const
{
  // Narrowed carrier: columns span only the *requested* input directions
  // (`_req_total_size`), so the composition's matmuls and the IFT solve's RHS
  // ("B matrix") are sized to the derivatives the user asked for. An input that
  // is never on the `in` side of a requested pair gets an all-zero block (it is
  // never differentiated, so it contributes nothing downstream).
  //
  // `batch_shape` is the FIRST input's full leading shape `(*dyn, *sub0)`. Each
  // dstate block carries the shared dynamic batch `*common_dyn` (with the first
  // input's own sub-batch axes stripped) and the per-variable folded storage
  // `sub_total_k * base_k`, so a heterogeneous mix of global and per-grain
  // inputs each gets the right shape (the first input being per-grain must not
  // impose its grain axis on a global input's sensitivity).
  const std::size_t in0_sub =
      _input_sub_batch_shapes.empty() ? 0 : _input_sub_batch_shapes[0].size();
  std::vector<int64_t> common_dyn(batch_shape.begin(),
                                  batch_shape.end() - static_cast<int64_t>(in0_sub));
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    int64_t sub_total = 1;
    for (auto s : _input_sub_batch_shapes[k])
      sub_total *= s;
    const int64_t folded = sub_total * _input_sizes[k];
    std::vector<int64_t> shape_vec = common_dyn;
    shape_vec.push_back(folded);
    shape_vec.push_back(_req_total_size);
    auto block = at::zeros(shape_vec, options);
    auto it = _req_input_offset.find(_input_names[k]);
    if (it != _req_input_offset.end())
    {
      // Requested inputs are plain-batch (the compile-time guard rejects
      // sub-batched requested pairs), so folded == _req_input_size here.
      const auto rs = _req_input_size.at(_input_names[k]);
      block.narrow(/*dim=*/-1, /*start=*/it->second, /*length=*/rs).copy_(at::eye(rs, options));
    }
    dstate[_input_names[k]] = block;
  }
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::_forward_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const
{
  const auto & seg = _segments.front();

  // Pack + validate caller inputs into a state map (canonical (*B, *base)).
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::jacobian: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

  // Run the jvp loader once: it returns (*outputs, *per-pair blocks) where the
  // blocks are row-major over seg.jacobian_pairs (the requested pairs).
  std::vector<at::Tensor> loader_in;
  loader_in.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
    loader_in.push_back(state.at(name).contiguous());
  // The jvp graph takes promoted parameters as per-batch inputs (schema v7), so
  // broadcast each stored parameter to the call batch before the call.
  const auto batch_shape =
      _input_names.empty() ? std::vector<int64_t>{} : _batch_shape_of(0, state.at(_input_names[0]));
  for (const auto & pname : seg.param_inputs)
    loader_in.push_back(
        broadcast_param_to_batch(_resolve_param(pname),
                                 batch_shape,
                                 static_cast<int64_t>(_param_base_shapes.at(pname).size())));
  const auto outs = seg.jvp_loader->run(loader_in);

  const std::size_t n_outs = seg.fwd_outputs.size();
  const std::size_t n_pairs = seg.jacobian_pairs.size();
  _assert(outs.size() == n_outs + n_pairs,
          "aoti::Model::jacobian: jvp loader returned ",
          outs.size(),
          " tensors, expected ",
          n_outs + n_pairs,
          ".");

  std::map<std::string, at::Tensor> outputs;
  for (std::size_t i = 0; i < n_outs; ++i)
    outputs[seg.fwd_outputs[i]] = outs[i];

  VariablePairJacobian jac;
  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & p = seg.jacobian_pairs[k];
    auto block = outs[n_outs + k]; // (*dyn, *in_sub, *out_base, *in_base)
    if (p.batch_independent)
    {
      // A batch-independent block traced to a static size-1 dynamic-batch axis;
      // drop the leading size-1 axes so it is returned unbatched (broadcasts
      // against any runtime batch). The metadata flag gates this, so a
      // state-dependent block that happens to run at batch=1 is never squeezed.
      const int64_t trail = static_cast<int64_t>(p.in_sub_batch_shape.size() +
                                                 p.out_base_shape.size() + p.in_base_shape.size());
      while (block.dim() > trail && block.size(0) == 1)
        block = block.squeeze(0);
    }
    jac[p.out_var][p.in_var] = block.contiguous();
  }
  return {std::move(outputs), std::move(jac)};
}

// Broadcast a stored promoted parameter `(*pbatch, *base)` to the call batch
// `(*batch, *base)` (right-aligned; `base` is the trailing `base_ndim` dims). See
// the declaration in internal.h. A scalar/unbatched parameter expands up to the
// batch; a per-batch-element parameter (`pbatch == batch`) passes through; any
// other `pbatch` is rejected by `expand` (the unsupported general-broadcast case).
at::Tensor
Model::Impl::broadcast_param_to_batch(const at::Tensor & param,
                                      const std::vector<int64_t> & batch,
                                      int64_t base_ndim)
{
  const auto & sizes = param.sizes();
  std::vector<int64_t> tgt(batch.begin(), batch.end());
  tgt.insert(tgt.end(), sizes.end() - base_ndim, sizes.end()); // (*batch, *base)
  return param.expand(tgt).contiguous();
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::_forward_param_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const
{
  const auto & seg = _segments.front();

  // Pack + validate caller inputs into a state map (canonical (*B, *base)).
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::param_jacobian: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

  // Runtime batch shape from the first structural input. The param-Jacobian graph
  // takes per-batch parameter inputs, so broadcast the stored scalar parameters
  // to this batch before the call (the value / jvp graphs keep them scalar).
  const auto batch_shape = _batch_shape_of(0, state.at(_input_names[0]));

  std::vector<at::Tensor> loader_in;
  loader_in.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
    loader_in.push_back(state.at(name).contiguous());
  for (const auto & pname : seg.param_inputs)
    loader_in.push_back(
        broadcast_param_to_batch(_resolve_param(pname),
                                 batch_shape,
                                 static_cast<int64_t>(_param_base_shapes.at(pname).size())));

  // The param-Jacobian graph returns ONLY the per-pair blocks (no value outputs),
  // in seg.param_jacobian_pairs order (outputs outer, params inner).
  const auto blocks = seg.param_jacobian_loader->run(loader_in);
  const std::size_t n_pairs = seg.param_jacobian_pairs.size();
  _assert(blocks.size() == n_pairs,
          "aoti::Model::param_jacobian: loader returned ",
          blocks.size(),
          " tensors, expected ",
          n_pairs,
          " (one block per (out, param) pair).");

  VariablePairJacobian pjac;
  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & p = seg.param_jacobian_pairs[k];
    pjac[p.out_var][p.param] = blocks[k].contiguous();
  }

  // Value outputs come from the value graph (the param-Jacobian graph emits only
  // derivative blocks). Run the segment's value loader on the same inputs.
  std::map<std::string, at::Tensor> out_state = state;
  _run_forward_segment(seg, out_state, batch_shape);
  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
    outputs[n] = out_state.at(n);

  return {std::move(outputs), std::move(pjac)};
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::_implicit_param_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const
{
  const auto & seg = _segments.front();

  // Pack + validate caller inputs into a state map (canonical (*B, *base)).
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::param_jacobian: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

  // Run the Newton solve to convergence; this writes the converged unknowns back
  // into `state` and hands us the converged per-group unknowns + per-group givens
  // (the exact tensors the ParamIFT graph was traced against).
  std::vector<at::Tensor> u_solved_groups;
  std::vector<at::Tensor> g_groups;
  _run_implicit_segment(seg, state, u_solved_groups, g_groups);

  // Runtime batch shape from the first converged unknown group: plain-batch =>
  // DENSE group tensor (*batch, group_storage), so batch = everything but the
  // last axis. (The implicit-promotion guard rejects sub-batched variables.)
  _assert(!u_solved_groups.empty(),
          "aoti::Model::param_jacobian: implicit segment produced no unknown groups.");
  const auto & u0 = u_solved_groups.front();
  std::vector<int64_t> batch_shape(u0.sizes().begin(), u0.sizes().end() - 1);

  // ParamIFT loader signature: (*u_groups, *g_groups, *params) where each param
  // is PER-BATCH. Broadcast the stored scalar parameters to the runtime batch.
  std::vector<at::Tensor> pift_in;
  pift_in.reserve(u_solved_groups.size() + g_groups.size() + seg.param_inputs.size());
  for (const auto & t : u_solved_groups)
    pift_in.push_back(t.contiguous());
  for (const auto & t : g_groups)
    pift_in.push_back(t.contiguous());
  for (const auto & pname : seg.param_inputs)
    pift_in.push_back(
        broadcast_param_to_batch(_resolve_param(pname),
                                 batch_shape,
                                 static_cast<int64_t>(_param_base_shapes.at(pname).size())));

  // One dense du/dθ block per (unknown, param) pair, in param_jacobian_pairs
  // order (unknowns outer, params inner).
  const auto blocks = seg.param_ift_loader->run(pift_in);
  const std::size_t n_pairs = seg.param_jacobian_pairs.size();
  _assert(blocks.size() == n_pairs,
          "aoti::Model::param_jacobian: ParamIFT loader returned ",
          blocks.size(),
          " blocks, expected ",
          n_pairs,
          " (one per (unknown, param) pair).");

  VariablePairJacobian pjac;
  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & p = seg.param_jacobian_pairs[k];
    pjac[p.out_var][p.param] = blocks[k].contiguous();
  }

  // Outputs are the converged unknowns (the implicit model's outputs).
  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
  {
    auto it = state.find(n);
    _assert(it != state.end(),
            "aoti::Model::param_jacobian: output '",
            n,
            "' was not produced by the implicit solve.");
    outputs[n] = it->second;
  }

  return {std::move(outputs), std::move(pjac)};
}

void
Model::Impl::_add_forward_param_direct(const Segment & seg,
                                       const std::map<std::string, at::Tensor> & state,
                                       const std::vector<int64_t> & common_dyn,
                                       std::map<std::string, at::Tensor> & dpstate) const
{
  // Run the forward param-Jacobian graph (structural inputs + promoted params
  // broadcast per-batch) and accumulate each d(out)/d(param) block at the
  // parameter's column band of dpstate[out].
  std::vector<at::Tensor> loader_in;
  loader_in.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
    loader_in.push_back(state.at(name).contiguous());
  for (const auto & pname : seg.param_inputs)
    loader_in.push_back(
        broadcast_param_to_batch(_resolve_param(pname),
                                 common_dyn,
                                 static_cast<int64_t>(_param_base_shapes.at(pname).size())));

  const auto blocks = seg.param_jacobian_loader->run(loader_in);
  _assert(blocks.size() == seg.param_jacobian_pairs.size(),
          "aoti::Model::param_jacobian: forward param-Jacobian loader returned ",
          blocks.size(),
          " blocks, expected ",
          seg.param_jacobian_pairs.size(),
          ".");

  for (std::size_t k = 0; k < seg.param_jacobian_pairs.size(); ++k)
  {
    const auto & pinfo = seg.param_jacobian_pairs[k];
    int64_t out_folded = 1;
    for (auto s : pinfo.out_base_shape)
      out_folded *= s;
    const int64_t psize = _req_param_size.at(pinfo.param);
    std::vector<int64_t> tgt = common_dyn;
    tgt.push_back(out_folded);
    tgt.push_back(psize);
    auto blk = blocks[k].reshape(tgt); // (*common_dyn, out_folded, psize)
    dpstate.at(pinfo.out_var)
        .narrow(/*dim=*/-1, /*start=*/_req_param_offset.at(pinfo.param), /*length=*/psize)
        .add_(blk);
  }
}

void
Model::Impl::_add_implicit_param_direct(const Segment & seg,
                                        const std::vector<at::Tensor> & u_solved_groups,
                                        const std::vector<at::Tensor> & g_groups,
                                        const std::vector<int64_t> & common_dyn,
                                        std::map<std::string, at::Tensor> & dpstate) const
{
  // Run the ParamIFT graph on the converged per-group unknowns + givens +
  // promoted params (broadcast per-batch) and accumulate each du/d(param) block
  // at the parameter's column band of dpstate[unknown].
  std::vector<at::Tensor> pift_in;
  pift_in.reserve(u_solved_groups.size() + g_groups.size() + seg.param_inputs.size());
  for (const auto & t : u_solved_groups)
    pift_in.push_back(t.contiguous());
  for (const auto & t : g_groups)
    pift_in.push_back(t.contiguous());
  for (const auto & pname : seg.param_inputs)
    pift_in.push_back(
        broadcast_param_to_batch(_resolve_param(pname),
                                 common_dyn,
                                 static_cast<int64_t>(_param_base_shapes.at(pname).size())));

  const auto blocks = seg.param_ift_loader->run(pift_in);
  _assert(blocks.size() == seg.param_jacobian_pairs.size(),
          "aoti::Model::param_jacobian: ParamIFT loader returned ",
          blocks.size(),
          " blocks, expected ",
          seg.param_jacobian_pairs.size(),
          ".");

  for (std::size_t k = 0; k < seg.param_jacobian_pairs.size(); ++k)
  {
    const auto & pinfo = seg.param_jacobian_pairs[k];
    int64_t u_folded = 1;
    for (auto s : pinfo.out_base_shape)
      u_folded *= s;
    const int64_t psize = _req_param_size.at(pinfo.param);
    std::vector<int64_t> tgt = common_dyn;
    tgt.push_back(u_folded);
    tgt.push_back(psize);
    auto blk = blocks[k].reshape(tgt); // (*common_dyn, u_folded, psize)
    dpstate.at(pinfo.out_var)
        .narrow(/*dim=*/-1, /*start=*/_req_param_offset.at(pinfo.param), /*length=*/psize)
        .add_(blk);
  }
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::_param_jacobian_dstate(const std::map<std::string, at::Tensor> & inputs) const
{
  // Pack + validate caller inputs (canonical (*B, *base)).
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::param_jacobian: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

  // Shared dynamic batch (input 0's leading shape minus its own sub-batch), the
  // same `common_dyn` the input-Jacobian carrier uses for its per-variable
  // dstate blocks.
  const auto & ref = state.at(_input_names.front());
  const auto batch_full = _batch_shape_of(0, ref);
  const std::size_t in0_sub =
      _input_sub_batch_shapes.empty() ? 0 : _input_sub_batch_shapes[0].size();
  std::vector<int64_t> common_dyn(batch_full.begin(),
                                  batch_full.end() - static_cast<int64_t>(in0_sub));
  int64_t common_numel = 1;
  for (auto s : common_dyn)
    common_numel *= s;

  // Seed dpstate = 0 for every master input (P parameter columns): inputs do not
  // depend on the promoted parameters; sensitivity accrues as segments inject
  // direct contributions and propagate them.
  const int64_t P = _param_total_size;
  std::map<std::string, at::Tensor> dpstate;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    int64_t sub_total = 1;
    for (auto s : _input_sub_batch_shapes[k])
      sub_total *= s;
    const int64_t folded = sub_total * _input_sizes[k];
    std::vector<int64_t> shape = common_dyn;
    shape.push_back(folded);
    shape.push_back(P);
    dpstate[_input_names[k]] = at::zeros(shape, ref.options());
  }

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      if (seg.jvp_loader)
        _run_forward_segment_jacobian(
            seg, state, dpstate, common_dyn); // indirect (+ advances state)
      else
      {
        _run_forward_segment(seg, state, common_dyn);
        for (const auto & oname : seg.fwd_outputs)
        {
          const auto & val = state.at(oname);
          const int64_t osz = common_numel > 0 ? val.numel() / common_numel : 0;
          std::vector<int64_t> shp = common_dyn;
          shp.push_back(osz);
          shp.push_back(P);
          dpstate[oname] = at::zeros(shp, ref.options());
        }
      }
      if (seg.param_jacobian_loader)
        _add_forward_param_direct(seg, state, common_dyn, dpstate); // direct
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      if (seg.ift_loader)
        _run_implicit_segment_jacobian(seg, u_solved_groups, g_groups, dpstate); // indirect
      else
        for (const auto & u : seg.unknowns)
        {
          std::vector<int64_t> shp = common_dyn;
          shp.push_back(u.var_size);
          shp.push_back(P);
          dpstate[u.name] = at::zeros(shp, ref.options());
        }
      if (seg.param_ift_loader)
        _add_implicit_param_direct(seg, u_solved_groups, g_groups, common_dyn, dpstate); // direct
    }
  }

  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
    outputs[n] = state.at(n);
  return {std::move(outputs), std::move(dpstate)};
}

void
Model::Impl::_run_forward_segment_jacobian(const Segment & seg,
                                           std::map<std::string, at::Tensor> & state,
                                           std::map<std::string, at::Tensor> & dstate,
                                           const std::vector<int64_t> & batch) const
{
  std::vector<at::Tensor> inputs;
  inputs.reserve(seg.fwd_inputs.size() + seg.param_inputs.size());
  for (const auto & name : seg.fwd_inputs)
  {
    auto it = state.find(name);
    _assert(it != state.end(),
            "aoti::Model: forward segment (JVP path) needs input '",
            name,
            "' which is not in the state map.");
    inputs.push_back(it->second.contiguous());
  }
  // Promoted-parameter tail: the jvp/jacobian value graph takes each parameter as
  // a per-batch input (schema v7), so broadcast the stored parameter to `batch`.
  for (const auto & pname : seg.param_inputs)
    inputs.push_back(broadcast_param_to_batch(
        _resolve_param(pname), batch, static_cast<int64_t>(_param_base_shapes.at(pname).size())));

  // JVP loader returns (*outputs..., *J_pairs) where J_pairs are one
  // per (out_var, in_var) pair, row-major (outputs outer, structural
  // inputs inner) -- matches seg.jacobian_pairs.
  const auto jvp_outs = seg.jvp_loader->run(inputs);
  const std::size_t n_outs = seg.fwd_outputs.size();
  const std::size_t n_pairs = seg.jacobian_pairs.size();
  _assert(jvp_outs.size() == n_outs + n_pairs,
          "aoti::Model: JVP loader returned ",
          jvp_outs.size(),
          " tensors, expected ",
          n_outs + n_pairs,
          " (",
          n_outs,
          " outputs + ",
          n_pairs,
          " Jacobian pairs).");
  for (std::size_t i = 0; i < n_outs; ++i)
    state[seg.fwd_outputs[i]] = jvp_outs[i];

  // Master_ndim from any dstate entry; all dstate tensors share their
  // trailing M dim.
  _assert(!seg.fwd_inputs.empty(),
          "aoti::Model: forward-Jacobian segment has no fwd_inputs; cannot infer master_ndim.");
  auto first_din_it = dstate.find(seg.fwd_inputs[0]);
  _assert(first_din_it != dstate.end(),
          "aoti::Model: dstate missing input '",
          seg.fwd_inputs[0],
          "' at Jacobian composition time.");
  const int64_t M = first_din_it->second.size(-1);

  // Per-output accumulator at (*B, out_var_size, M). Batch shape from
  // dstate[fwd_inputs[0]] (which is the canonical (*B, in_var_size, M)).
  std::vector<int64_t> batch_shape;
  {
    const auto & ref = first_din_it->second;
    for (int64_t d = 0; d < ref.dim() - 2; ++d)
      batch_shape.push_back(ref.size(d));
  }
  std::map<std::string, at::Tensor> dout_acc;
  // Cache per-output var_size from the JVP value tensor (its trailing
  // shape matches the output's natural (*sub, *base)).
  for (std::size_t i = 0; i < n_outs; ++i)
  {
    const auto & val = jvp_outs[i];
    int64_t out_var_size = 1;
    for (int64_t d = static_cast<int64_t>(batch_shape.size()); d < val.dim(); ++d)
      out_var_size *= val.size(d);
    std::vector<int64_t> shape = batch_shape;
    shape.push_back(out_var_size);
    shape.push_back(M);
    dout_acc[seg.fwd_outputs[i]] = at::zeros(shape, val.options());
  }

  // Iterate pairs; per-pair matmul + reshape + accumulate.
  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & pair = jvp_outs[n_outs + k];
    const auto & pinfo = seg.jacobian_pairs[k];

    int64_t in_base_total = 1;
    for (auto s : pinfo.in_base_shape)
      in_base_total *= s;
    int64_t in_sub_total = 1;
    for (auto s : pinfo.in_sub_batch_shape)
      in_sub_total *= s;
    int64_t out_base_total = 1;
    for (auto s : pinfo.out_base_shape)
      out_base_total *= s;
    const int64_t in_var_size = in_sub_total * in_base_total;

    // dstate[in_var] shape: (*B, in_var_size, M). View as (*B, *in_sub,
    // *in_base, M) -- last 1+in_base_ndim dims index over the in side.
    auto din_it = dstate.find(pinfo.in_var);
    _assert(din_it != dstate.end(),
            "aoti::Model: forward-Jacobian needs dstate['",
            pinfo.in_var,
            "'] which is missing.");
    const auto & din = din_it->second;
    std::vector<int64_t> din_target = batch_shape;
    for (auto s : pinfo.in_sub_batch_shape)
      din_target.push_back(s);
    for (auto s : pinfo.in_base_shape)
      din_target.push_back(s);
    din_target.push_back(M);
    auto din_view = din.reshape(din_target);

    // pair shape: (*B, *sub, *out_base, *in_base). View as (*B, *sub,
    // out_base_total, in_base_total) for the matmul, where *sub may be
    // empty (in_sub absent) or in_sub (chain-rule preserved). For
    // simplicity, flatten the pair's trailing 2*ndim into (out_base,
    // in_base) and rely on the matmul to broadcast against din_view.
    //
    // Tensordot over in_base axes. Use reshape to (..., out_base,
    // in_base) then matmul against (..., in_base, M).
    const int64_t pair_trail =
        static_cast<int64_t>(pinfo.out_base_shape.size() + pinfo.in_base_shape.size());
    _assert(pair.dim() >= pair_trail,
            "aoti::Model: Jacobian pair tensor ndim=",
            pair.dim(),
            " < expected trail=",
            pair_trail);
    const int64_t pair_lead = pair.dim() - pair_trail;
    std::vector<int64_t> pair_shape;
    pair_shape.reserve(static_cast<std::size_t>(pair_lead + 2));
    for (int64_t d = 0; d < pair_lead; ++d)
      pair_shape.push_back(pair.size(d));
    pair_shape.push_back(out_base_total);
    pair_shape.push_back(in_base_total);
    auto pair_view = pair.reshape(pair_shape);

    // din_view's trailing layout: (..., in_base_total, M). Reshape to
    // collapse in_base axes and (if present) in_sub axes.
    std::vector<int64_t> din_collapsed_shape = batch_shape;
    if (!pinfo.in_sub_batch_shape.empty())
    {
      for (auto s : pinfo.in_sub_batch_shape)
        din_collapsed_shape.push_back(s);
    }
    din_collapsed_shape.push_back(in_base_total);
    din_collapsed_shape.push_back(M);
    auto din_collapsed = din.reshape(din_collapsed_shape);

    // pair_view @ din_collapsed: pair_view (*B, *sub_or_none,
    // out_base_total, in_base_total) and din_collapsed
    // (*B, *in_sub, in_base_total, M). For paired BLOCK pairs (sub
    // present on both), broadcast aligns the sub. For DENSE in_var
    // (no in_sub), din shape (*B, in_base_total, M).
    auto contrib = at::matmul(pair_view, din_collapsed);
    // contrib shape (*B, *sub_union, out_base_total, M); for the DENSE
    // out_var case, accumulate as-is into (*B, out_var_size, M).
    // For BLOCK out_var (out_sub present), the sub axes need to be
    // folded into out_var_size. The current implementation supports
    // only DENSE out_var; reject BLOCK out_var here pending the future
    // BLOCK-output forward-Jacobian path.
    const int64_t accumulator_var_size = dout_acc[pinfo.out_var].size(-2);
    int64_t contrib_var_size = 1;
    for (int64_t d = static_cast<int64_t>(batch_shape.size()); d < contrib.dim() - 1; ++d)
      contrib_var_size *= contrib.size(d);
    if (contrib_var_size == accumulator_var_size)
    {
      std::vector<int64_t> flat_shape = batch_shape;
      flat_shape.push_back(accumulator_var_size);
      flat_shape.push_back(M);
      dout_acc[pinfo.out_var] = dout_acc[pinfo.out_var] + contrib.reshape(flat_shape);
    }
    else
    {
      _assert(false,
              "aoti::Model: forward-Jacobian per-pair contribution for (",
              pinfo.out_var,
              ", ",
              pinfo.in_var,
              ") has var_size ",
              contrib_var_size,
              " which does not match accumulator ",
              accumulator_var_size,
              ". BLOCK-output forward-Jacobian path is not yet implemented.");
    }
    (void)in_var_size;
  }

  for (const auto & oname : seg.fwd_outputs)
    dstate[oname] = dout_acc[oname].contiguous();
}

void
Model::Impl::_run_implicit_segment_jacobian(const Segment & seg,
                                            const std::vector<at::Tensor> & u_solved_groups,
                                            const std::vector<at::Tensor> & g_groups,
                                            std::map<std::string, at::Tensor> & dstate) const
{
  // IFT loader signature: (*u_groups, *g_groups, [params...]) -> *blocks, one
  // per (unknown, given) pair in seg.jacobian_pairs order (the disassemble of
  // -du/dg). Compose each block against dstate[given] into dstate[unknown] --
  // the same per-pair Jacobian path a forward segment uses.
  std::vector<at::Tensor> ift_in;
  ift_in.reserve(u_solved_groups.size() + g_groups.size() + seg.param_inputs.size());
  for (const auto & t : u_solved_groups)
    ift_in.push_back(t.contiguous());
  for (const auto & t : g_groups)
    ift_in.push_back(t.contiguous());
  for (auto & p : _gather_params(seg.param_inputs))
    ift_in.push_back(std::move(p));

  const auto blocks = seg.ift_loader->run(ift_in);
  const std::size_t n_pairs = seg.jacobian_pairs.size();
  _assert(blocks.size() == n_pairs,
          "aoti::Model: IFT loader returned ",
          blocks.size(),
          " blocks, expected ",
          n_pairs,
          " (one per (unknown, given) pair).");

  // Master dim M + batch shape from any given's dstate (all share trailing M).
  _assert(!seg.givens.empty(),
          "aoti::Model: implicit-segment IFT has no givens; cannot infer master_ndim.");
  auto first_dg_it = dstate.find(seg.givens.front().name);
  _assert(first_dg_it != dstate.end(),
          "aoti::Model: dstate missing given '",
          seg.givens.front().name,
          "' at IFT composition time.");
  const auto & dg_ref = first_dg_it->second;
  const int64_t M = dg_ref.size(-1);
  std::vector<int64_t> batch_shape;
  for (int64_t d = 0; d < dg_ref.dim() - 2; ++d)
    batch_shape.push_back(dg_ref.size(d));

  // Per-unknown accumulator (*B, u.var_size, M). Unknowns are not seeded in
  // _init_dstate (they are produced by composition), so start at zero.
  std::map<std::string, at::Tensor> du_acc;
  for (const auto & u : seg.unknowns)
  {
    std::vector<int64_t> shape = batch_shape;
    shape.push_back(u.var_size);
    shape.push_back(M);
    du_acc[u.name] = at::zeros(shape, dg_ref.options());
  }

  for (std::size_t k = 0; k < n_pairs; ++k)
  {
    const auto & blk = blocks[k];
    const auto & pinfo = seg.jacobian_pairs[k];

    auto din_it = dstate.find(pinfo.in_var);
    _assert(din_it != dstate.end(),
            "aoti::Model: IFT needs dstate['",
            pinfo.in_var,
            "'] which is missing.");
    const auto & din = din_it->second;

    // ``disassemble`` blocks carry exactly two trailing storage axes
    // ``(*B, [sub], out_storage, in_storage)`` (out_storage == unknown var_size,
    // in_storage == given var_size). For a plain batch the block is
    // ``(*B, out_storage, in_storage)`` and ``dstate[given]`` is
    // ``(*B, in_storage, M)``, so the IFT composition is a direct matmul. A
    // BLOCK (per-grain) side adds an intermediate sub axis that needs the paired
    // / cross-grain reduction -- deferred (rejected) until the sub-batch IFT
    // path lands.
    _assert(pinfo.in_sub_batch_shape.empty() &&
                blk.dim() == static_cast<int64_t>(batch_shape.size()) + 2,
            "aoti::Model: sub-batch (BLOCK) IFT Jacobian for (",
            pinfo.out_var,
            ", ",
            pinfo.in_var,
            ") is not yet implemented.");

    auto contrib = at::matmul(blk, din); // (*B, out_storage, M)
    du_acc[pinfo.out_var] = du_acc[pinfo.out_var] + contrib;
  }

  // Negation already applied Python-side (IFT emits -du/dg).
  for (const auto & u : seg.unknowns)
    dstate[u.name] = du_acc[u.name].contiguous();
}

} // namespace neml2::aoti
