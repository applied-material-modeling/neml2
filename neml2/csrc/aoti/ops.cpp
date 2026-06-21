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

std::map<std::string, at::Tensor>
Model::Impl::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::forward: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      _run_forward_segment(seg, state);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
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
  // Pack state from caller inputs (validating each is canonical (*B, *base)).
  std::map<std::string, at::Tensor> state;
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::jacobian: missing required input '", n, "'.");
    _validate_input_shape(k, it->second);
    state[n] = it->second.contiguous();
  }

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
        _run_forward_segment_jacobian(seg, state, dstate);
      else
      {
        // Off-path forward segment (pruned by `-d` selection): its outputs are
        // on no requested derivative path, so advance `state` via the value
        // graph and zero-fill its outputs' dstate. The zeros are never consumed
        // by a kept pair -- only discarded in the final per-pair slice.
        _run_forward_segment(seg, state);
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
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
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
Model::Impl::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
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

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::jvp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & tangents) const
{
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
