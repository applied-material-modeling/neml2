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
// Public ops: forward / jvp / jacobian
// ----------------------------------------------------------------------------
// Compose the per-segment runners (defined in solve.cpp and jacobian.cpp) into
// the model's three public entry points. `Model` forwards onto these via Impl.

#include "neml2/csrc/aoti/internal.h"

// at::infer_size (broadcast two shapes) for the common-batch computation.
#include <ATen/ExpandUtils.h>

// param_vjp runs an AOTI loader directly (the other ops delegate loader calls to
// jacobian.cpp); pull in the full loader type for ->run().
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace neml2::aoti
{
void
Model::Impl::_validate_input_shape(std::size_t idx, const at::Tensor & t) const
{
  const auto & base = _input_base_shapes[idx];
  const auto & name = _input_names[idx];
  const at::IntArrayRef base_ref(base);
  const int64_t bn = static_cast<int64_t>(base.size());
  _assert(t.dim() >= bn,
          "aoti::Model: input '",
          name,
          "' has ",
          t.dim(),
          " dim(s) but its base shape ",
          base_ref,
          " requires at least ",
          bn,
          " trailing axes (canonical input shape is (*B, *base)).");
  // The trailing base_ndim axes must equal the declared base shape exactly, so
  // e.g. an SR2 must arrive as (*B, 6) -- not (*B, 1) or (*B, 3, 2). (A Scalar
  // has an empty base shape, so any leading shape is a valid batch.)
  for (int64_t i = 0; i < bn; ++i)
  {
    const int64_t got = t.size(t.dim() - bn + i);
    _assert(got == base[i],
            "aoti::Model: input '",
            name,
            "' has non-canonical shape ",
            t.sizes(),
            "; expected trailing base shape ",
            base_ref,
            " (mismatch at base axis ",
            i,
            ": got ",
            got,
            ", expected ",
            base[i],
            ").");
  }
}

std::vector<int64_t>
Model::Impl::_batch_shape_of(std::size_t idx, const at::Tensor & t) const
{
  const int64_t bn = static_cast<int64_t>(_input_base_shapes[idx].size());
  std::vector<int64_t> batch;
  for (int64_t d = 0; d < t.dim() - bn; ++d)
    batch.push_back(t.size(d));
  return batch;
}

std::vector<int64_t>
Model::Impl::_dynamic_batch_shape_of(std::size_t idx, const at::Tensor & t) const
{
  // Strip BOTH the sub-batch axes and the trailing base axes; what remains is
  // the leading plain (dynamic) batch. `_input_sub_batch_shapes` is populated
  // per input at construction (an empty vector for a plain-batch input, and
  // empty overall for an older cache without the metadata -> sub_ndim 0).
  const int64_t sub_ndim = _input_sub_batch_shapes.empty()
                               ? 0
                               : static_cast<int64_t>(_input_sub_batch_shapes[idx].size());
  const int64_t trailing = static_cast<int64_t>(_input_base_shapes[idx].size()) + sub_ndim;
  std::vector<int64_t> dyn;
  for (int64_t d = 0; d < t.dim() - trailing; ++d)
    dyn.push_back(t.size(d));
  return dyn;
}

std::map<std::string, at::Tensor>
Model::Impl::_prepare_inputs(const std::map<std::string, at::Tensor> & inputs) const
{
  std::map<std::string, at::Tensor> state;
  // First pass: validate + materialize each input, accumulating the common
  // DYNAMIC (plain) batch across all of them. Only the plain batch is unified:
  // a sub-batched input's per-site axes (crystal-plasticity per-grain /
  // per-slip) are structural and are stripped alongside the base axes before
  // broadcasting -- unifying them would collide a global input's (B,) against a
  // per-grain input's (B, ngrain). This mirrors the typed routes, whose
  // broadcast_to_common_batch broadcasts only the dynamic batch per its
  // per-input sub_batch_ndim.
  std::vector<int64_t> dyn;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
    dyn = at::infer_size(dyn, _dynamic_batch_shape_of(k, state[n]));
  }
  // Second pass: lift every input's dynamic batch to the common shape, leaving
  // its sub-batch and base axes untouched -- a batch-independent input (e.g. a
  // scalar TIME force) is broadcast to the call batch without disturbing a
  // sub-batched input's per-grain axes.
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    const at::Tensor & t = state[n];
    const int64_t sub_ndim = _input_sub_batch_shapes.empty()
                                 ? 0
                                 : static_cast<int64_t>(_input_sub_batch_shapes[k].size());
    const int64_t keep = static_cast<int64_t>(_input_base_shapes[k].size()) + sub_ndim;
    // target = (*common_dyn, *sub_k, *base_k): common plain batch, then this
    // input's own trailing sub-batch + base axes verbatim.
    std::vector<int64_t> target(dyn);
    for (int64_t d = t.dim() - keep; d < t.dim(); ++d)
      target.push_back(t.size(d));
    state[n] = t.broadcast_to(target).contiguous();
  }
  return state;
}

std::map<std::string, at::Tensor>
Model::Impl::forward(const std::map<std::string, at::Tensor> & inputs,
                     const std::map<std::string, at::Tensor> & param_overrides) const
{
  const ParamOverrideGuard _pog(this, param_overrides);
  auto state = _prepare_inputs(inputs);

  // Call batch from the first structural input (its base stripped). Forward
  // segments broadcast each promoted parameter to this batch before the call,
  // since the value graphs take parameters as per-batch inputs.
  const std::vector<int64_t> batch =
      _input_names.empty() ? std::vector<int64_t>{} : _batch_shape_of(0, state.at(_input_names[0]));

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      _run_forward_segment(seg, state, batch);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      if (seg.max_substepping_level > 0)
      {
        if (_masking_ok(seg, state))
          _run_implicit_segment_substepped_masked(seg, state);
        else
          _run_implicit_segment_substepped(seg, state, u_solved_groups, g_groups);
      }
      else
        _run_implicit_segment(seg, state, u_solved_groups, g_groups);
    }
  }

  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
  {
    auto it = state.find(n);
    _assert(it != state.end(),
            "aoti::Model::forward: output '",
            n,
            "' was not produced by any segment.");
    outputs[n] = it->second;
  }
  return outputs;
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::_jacobian_dstate(const std::map<std::string, at::Tensor> & inputs) const
{
  // Pack state from caller inputs (validating + broadcasting to the common batch).
  auto state = _prepare_inputs(inputs);

  // dstate[var] is (*B, var_size, M) where M = _input_total_size. Initialized
  // with identity columns for each master input, then composed segment-by-
  // segment in lockstep with `state`. The batch shape is the leading axes of a
  // master input ahead of its trailing base axes (validated above). For a
  // sub-batched master input the (*dyn, *sub) region is all treated as batch,
  // matching the legacy var_size-trailing split.
  const auto & ref = state.at(_input_names.front());
  const auto batch_shape_vec = _batch_shape_of(0, ref);
  at::IntArrayRef batch_shape(batch_shape_vec);

  std::map<std::string, at::Tensor> dstate;
  _init_dstate(ref.options(), batch_shape, dstate);

  int64_t batch_numel = 1;
  for (auto s : batch_shape_vec)
    batch_numel *= s;
  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      if (seg.jvp_loader)
        _run_forward_segment_jacobian(seg, state, dstate, batch_shape_vec);
      else
      {
        // Off-path forward segment (pruned by `-d` selection): its outputs are
        // on no requested derivative path, so advance `state` via the value
        // graph and zero-fill its outputs' dstate. The zeros are never consumed
        // by a kept pair -- only discarded in the final per-pair slice.
        _run_forward_segment(seg, state, batch_shape_vec);
        for (const auto & oname : seg.fwd_outputs)
        {
          const auto & val = state.at(oname);
          const int64_t osz = batch_numel > 0 ? val.numel() / batch_numel : 0;
          std::vector<int64_t> shp = batch_shape_vec;
          shp.push_back(osz);
          shp.push_back(_req_total_size);
          dstate[oname] = at::zeros(shp, ref.options());
        }
      }
    }
    else if (seg.max_substepping_level > 0 && seg.ift_loader)
    {
      // Substepped solve + chained consistent-tangent accumulation in one
      // bisection recursion (state + dstate advanced together). Masked when the
      // dynamic batch is 1-D so only the still-unconverged rows are re-solved.
      if (_masking_ok(seg, state))
        _run_implicit_segment_substepped_masked_jacobian(seg, state, dstate);
      else
        _run_implicit_segment_substepped_jacobian(seg, state, dstate);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      if (seg.max_substepping_level > 0)
        _run_implicit_segment_substepped(seg, state, u_solved_groups, g_groups);
      else
        _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      if (seg.ift_loader)
        _run_implicit_segment_jacobian(seg, u_solved_groups, g_groups, dstate);
      else
        // Off-path implicit segment: forward solve already advanced `state`;
        // zero-fill the unknowns' dstate (never consumed by a kept pair).
        for (const auto & u : seg.unknowns)
        {
          std::vector<int64_t> shp = batch_shape_vec;
          shp.push_back(u.var_size);
          shp.push_back(_req_total_size);
          dstate[u.name] = at::zeros(shp, ref.options());
        }
    }
  }

  // Pack the output values; the caller slices `dstate[out]` per requested pair.
  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
    outputs[n] = state.at(n);

  return {std::move(outputs), std::move(dstate)};
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::jacobian(const std::map<std::string, at::Tensor> & inputs,
                      const std::map<std::string, at::Tensor> & param_overrides) const
{
  const ParamOverrideGuard _pog(this, param_overrides);
  _assert(!_derivatives.empty(),
          "aoti::Model::jacobian: this artifact was compiled with no derivative graphs. "
          "Recompile with `neml2-compile -d OUT:IN` (e.g. `-d :` for all pairs).");

  // Single forward segment: return the compiled per-pair blocks directly (no
  // dense flat-J round-trip), so a batch-independent block (e.g. a constant
  // stiffness tensor) is returned unbatched at its natural (*out_base, *in_base).
  if (_is_single_forward_jac())
    return _forward_pair_blocks(inputs);

  auto [outputs, dstate] = _jacobian_dstate(inputs); // dstate[var]: (*B, var_folded, M_req)

  std::map<std::string, std::size_t> out_idx, in_idx;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
    out_idx[_output_names[i]] = i;
  for (std::size_t j = 0; j < _input_names.size(); ++j)
    in_idx[_input_names[j]] = j;

  // Slice each requested output's dstate block directly (no flat cat): take the
  // input's column band [req_offset, +in_var) and reshape to
  // (*B, *out_base, *in_base). Per-output slicing keeps offsets correct even
  // with heterogeneous folded output sizes.
  VariablePairJacobian jac;
  for (const auto & [o, i] : _derivatives)
  {
    const auto & blk = dstate.at(o); // (*B, out_folded, M_req)
    std::vector<int64_t> batch(blk.sizes().begin(), blk.sizes().end() - 2);
    const auto ji = in_idx.at(i);
    auto col = blk.narrow(-1, _req_input_offset.at(i), _req_input_size.at(i));
    std::vector<int64_t> block_shape = batch;
    block_shape.insert(block_shape.end(),
                       _output_base_shapes[out_idx.at(o)].begin(),
                       _output_base_shapes[out_idx.at(o)].end());
    block_shape.insert(
        block_shape.end(), _input_base_shapes[ji].begin(), _input_base_shapes[ji].end());
    jac[o][i] = col.reshape(block_shape).contiguous();
  }

  return {std::move(outputs), std::move(jac)};
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::param_jacobian(const std::map<std::string, at::Tensor> & inputs,
                            const std::map<std::string, at::Tensor> & param_overrides) const
{
  const ParamOverrideGuard _pog(this, param_overrides);
  _assert(!_param_derivatives.empty(),
          "aoti::Model::param_jacobian: this artifact was compiled with no parameter "
          "derivatives. Recompile with `neml2-compile -p PARAM -d OUT:PARAM`.");
  // Two single-segment fast paths return the param blocks directly: a forward
  // segment carrying the dense param-Jacobian loader, and an implicit segment
  // carrying the ParamIFT loader (du/dθ = -A⁻¹ ∂r/∂θ).
  if (_segments.size() == 1 && _segments.front().kind == SegmentKind::Forward &&
      static_cast<bool>(_segments.front().param_jacobian_loader))
    return _forward_param_pair_blocks(inputs);
  if (_segments.size() == 1 && _segments.front().kind == SegmentKind::Implicit &&
      static_cast<bool>(_segments.front().param_ift_loader))
    return _implicit_param_pair_blocks(inputs);

  // Multi-segment (composed) artifacts: compose the parameter sensitivity across
  // segments through the zero-seeded parameter carrier, then slice each
  // requested (out, param) block from dpstate[out]'s parameter column band.
  auto [outputs, dpstate] = _param_jacobian_dstate(inputs); // dpstate[var]: (*B, var_folded, P)

  std::map<std::string, std::size_t> out_idx;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
    out_idx[_output_names[i]] = i;

  VariablePairJacobian pjac;
  for (const auto & [o, p] : _param_derivatives)
  {
    const auto & blk = dpstate.at(o); // (*B, out_folded, P)
    std::vector<int64_t> batch(blk.sizes().begin(), blk.sizes().end() - 2);
    auto col = blk.narrow(/*dim=*/-1, _req_param_offset.at(p), _req_param_size.at(p));
    std::vector<int64_t> block_shape = batch;
    block_shape.insert(block_shape.end(),
                       _output_base_shapes[out_idx.at(o)].begin(),
                       _output_base_shapes[out_idx.at(o)].end());
    // Trailing param axes are the NATURAL base (param_base_shape), not the stored
    // parameter's full shape -- a batched parameter's batch dim is carried by the
    // block's leading `batch`, never injected into the param-base axes.
    const auto & pbase = _param_base_shapes.at(p);
    block_shape.insert(block_shape.end(), pbase.begin(), pbase.end());
    pjac[o][p] = col.reshape(block_shape).contiguous();
  }
  return {std::move(outputs), std::move(pjac)};
}

std::map<std::string, at::Tensor>
Model::Impl::param_vjp(const std::map<std::string, at::Tensor> & inputs,
                       const std::map<std::string, at::Tensor> & cotangents,
                       const std::map<std::string, at::Tensor> & param_overrides) const
{
  const ParamOverrideGuard _pog(this, param_overrides);
  _assert(!_param_derivatives.empty(),
          "aoti::Model::param_vjp: this artifact was compiled with no parameter derivatives. "
          "Recompile with `neml2-compile -p PARAM -d OUT:PARAM`.");

  // Collapse a per-batch-element adjoint `(*call_batch, *param_base)` to match the
  // parameter's stored shape, mirroring the eager param_vjp (whose autograd leaf
  // takes the stored shape): keep per-element for a BATCHED parameter, sum over
  // the batch for an unbatched (GLOBAL) one. The per-batch adjoint graph + this
  // collapse are numerically identical to the old scalar-leaf graph for a global
  // parameter (sum-of-per-element == the scalar-leaf gradient).
  auto collapse = [&](const std::string & p, const at::Tensor & per_elem) -> at::Tensor
  {
    auto bit = _param_base_shapes.find(p);
    const int64_t base_ndim =
        (bit != _param_base_shapes.end()) ? static_cast<int64_t>(bit->second.size()) : 0;
    const bool batched = (_resolve_param(p).dim() - base_ndim) >= 1;
    if (batched)
      return per_elem.contiguous();
    const int64_t nlead = per_elem.dim() - base_ndim;
    if (nlead <= 0)
      return per_elem.contiguous();
    std::vector<int64_t> dims;
    dims.reserve(static_cast<std::size_t>(nlead));
    for (int64_t d = 0; d < nlead; ++d)
      dims.push_back(d);
    return per_elem.sum(dims).contiguous();
  };

  // Single forward segment: a dedicated adjoint graph computes the per-batch
  // adjoint in one reverse pass -- the cheapest form for many params.
  const auto & seg = _segments.front();
  if (_segments.size() == 1 && seg.kind == SegmentKind::Forward &&
      static_cast<bool>(seg.param_vjp_loader))
  {
    // Validate + pack structural inputs (canonical (*B, *base)), broadcasting to
    // the common batch.
    auto state = _prepare_inputs(inputs);

    // Loader inputs: structural state, promoted parameters broadcast PER-BATCH
    // (the adjoint graph differentiates a per-batch leaf -> per-batch-element
    // gradient), then one cotangent per output in param_vjp_outputs order (each at
    // the output's natural (*B, *out_base) shape).
    const std::vector<int64_t> batch = _input_names.empty()
                                           ? std::vector<int64_t>{}
                                           : _batch_shape_of(0, state.at(_input_names[0]));
    std::vector<at::Tensor> loader_in;
    loader_in.reserve(seg.fwd_inputs.size() + seg.param_inputs.size() +
                      seg.param_vjp_outputs.size());
    for (const auto & name : seg.fwd_inputs)
      loader_in.push_back(state.at(name).contiguous());
    for (const auto & pname : seg.param_inputs)
      loader_in.push_back(broadcast_param_to_batch(
          _resolve_param(pname), batch, static_cast<int64_t>(_param_base_shapes.at(pname).size())));
    for (const auto & oname : seg.param_vjp_outputs)
    {
      auto it = cotangents.find(oname);
      _assert(it != cotangents.end(),
              "aoti::Model::param_vjp: missing cotangent for output '",
              oname,
              "'.");
      loader_in.push_back(it->second.contiguous());
    }

    // One per-batch-element gradient per parameter, in param_vjp_params order;
    // collapse each to the stored parameter's shape (sum global / keep batched).
    const auto grads = seg.param_vjp_loader->run(loader_in);
    _assert(grads.size() == seg.param_vjp_params.size(),
            "aoti::Model::param_vjp: VJP loader returned ",
            grads.size(),
            " gradients, expected ",
            seg.param_vjp_params.size(),
            ".");

    std::map<std::string, at::Tensor> out;
    for (std::size_t k = 0; k < grads.size(); ++k)
      out[seg.param_vjp_params[k]] = collapse(seg.param_vjp_params[k], grads[k]);
    return out;
  }

  // Implicit / composed: there is no dedicated adjoint graph, but the parameter
  // sensitivity carrier already builds the per-element d(out)/d(param) in O(n_out)
  // reverse passes (independent of the parameter count), so the VJP is its
  // contraction with the output cotangents over the OUTPUT-BASE axes only:
  //     adjoint_p[*B] = Σ_o <w_o[*B], d(out_o)/d(θ_p)[*B]>   (per batch element),
  // then collapsed (summed over batch for a global parameter).
  auto [outputs, pjac] = param_jacobian(inputs);
  (void)outputs;

  std::map<std::string, int64_t> out_base_ndim;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
    out_base_ndim[_output_names[i]] = static_cast<int64_t>(_output_base_shapes[i].size());

  std::map<std::string, at::Tensor> grads;
  for (const auto & [o, p] : _param_derivatives)
  {
    auto cit = cotangents.find(o);
    _assert(cit != cotangents.end(),
            "aoti::Model::param_vjp: missing cotangent for output '",
            o,
            "' (a (",
            o,
            ", ",
            p,
            ") parameter derivative was compiled, so its cotangent is required).");
    const auto & w = cit->second;      // (*B, *out_base)
    const auto & J = pjac.at(o).at(p); // (*B, *out_base, *param_base)
    const int64_t ob = out_base_ndim.at(o);
    const int64_t bnd = w.dim() - ob; // batch ndim
    _assert(bnd >= 0 && J.dim() == w.dim() + static_cast<int64_t>(_param_base_shapes.at(p).size()),
            "aoti::Model::param_vjp: cotangent for '",
            o,
            "' (shape ",
            w.sizes(),
            ") is incompatible with the Jacobian block (",
            o,
            ", ",
            p,
            ") shape ",
            J.sizes(),
            ".");
    std::vector<int64_t> batch_sizes(w.sizes().begin(), w.sizes().begin() + bnd);
    int64_t bn = 1;
    for (auto s : batch_sizes)
      bn *= s;
    int64_t obn = 1;
    for (int64_t d = bnd; d < w.dim(); ++d)
      obn *= w.size(d);
    const auto & pbase = _param_base_shapes.at(p);
    int64_t pb = 1;
    for (auto s : pbase)
      pb *= s;
    // Contract the out_base axes (keep batch): (bn, obn, 1) * (bn, obn, pb) -> (bn, pb).
    auto per_elem = (w.reshape({bn, obn, 1}) * J.reshape({bn, obn, pb})).sum(1);
    std::vector<int64_t> pe_shape = batch_sizes;
    pe_shape.insert(pe_shape.end(), pbase.begin(), pbase.end());
    auto contrib = collapse(p, per_elem.reshape(pe_shape)); // (*B,*pbase) or (*pbase)
    auto git = grads.find(p);
    grads[p] = (git == grads.end()) ? contrib : (git->second + contrib).contiguous();
  }
  return grads;
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::jvp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & tangents,
                 const std::map<std::string, at::Tensor> & param_overrides) const
{
  const ParamOverrideGuard _pog(this, param_overrides);
  _assert(!_derivatives.empty(),
          "aoti::Model::jvp: this artifact was compiled with no derivative graphs. "
          "Recompile with `neml2-compile -d OUT:IN` (e.g. `-d :` for all pairs).");

  // Per covered output, contract its requested input column bands with the
  // matching tangents. Outputs in no requested pair are omitted; inputs not
  // paired with an output contribute nothing (they were not differentiated).
  auto [outputs, dstate] = _jacobian_dstate(inputs); // dstate[var]: (*B, var_folded, M_req)

  std::map<std::string, std::size_t> out_idx, in_idx;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
    out_idx[_output_names[i]] = i;
  for (std::size_t j = 0; j < _input_names.size(); ++j)
    in_idx[_input_names[j]] = j;

  // Per-input tangent column (*B, in_var_size, 1) at the given batch; a missing
  // tangent is zero.
  auto tangent_col = [&](const std::string & iname,
                         const std::vector<int64_t> & batch,
                         const at::TensorOptions & opts) -> at::Tensor
  {
    const auto ji = in_idx.at(iname);
    const auto sz = _input_sizes[ji];
    auto it = tangents.find(iname);
    if (it != tangents.end())
    {
      _validate_input_shape(ji, it->second); // canonical (*B, *base) contract
      auto shape = _batch_shape_of(ji, it->second);
      shape.push_back(sz); // flatten trailing base axes into the var_size slot
      return it->second.reshape(shape).to(_dtype).unsqueeze(-1).contiguous();
    }
    std::vector<int64_t> shape = batch;
    shape.push_back(sz);
    shape.push_back(1);
    return at::zeros(shape, opts);
  };

  std::map<std::string, at::Tensor> jvp_outputs;
  for (const auto & [o, ins] : _deriv_by_out)
  {
    const auto & blk = dstate.at(o); // (*B, out_folded, M_req)
    std::vector<int64_t> batch(blk.sizes().begin(), blk.sizes().end() - 2);
    at::Tensor acc;
    for (const auto & iname : ins)
    {
      auto col = blk.narrow(-1, _req_input_offset.at(iname), _req_input_size.at(iname));
      auto contrib = at::matmul(col, tangent_col(iname, batch, blk.options())).squeeze(-1);
      acc = acc.defined() ? acc + contrib : contrib;
    }
    std::vector<int64_t> shape = batch;
    shape.insert(shape.end(),
                 _output_base_shapes[out_idx.at(o)].begin(),
                 _output_base_shapes[out_idx.at(o)].end());
    jvp_outputs[o] = acc.reshape(shape).contiguous();
  }

  return {std::move(outputs), std::move(jvp_outputs)};
}

} // namespace neml2::aoti
