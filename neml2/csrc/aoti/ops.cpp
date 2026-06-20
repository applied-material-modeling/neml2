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

std::pair<std::map<std::string, at::Tensor>, at::Tensor>
Model::Impl::_jacobian_flat(const std::map<std::string, at::Tensor> & inputs) const
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

  for (const auto & seg : _segments)
  {
    if (seg.kind == SegmentKind::Forward)
    {
      _assert(static_cast<bool>(seg.jvp_loader),
              "aoti::Model::jacobian: forward segment is missing its jvp_package loader. "
              "Re-export via `neml2-compile`.");
      _run_forward_segment_jacobian(seg, state, dstate);
    }
    else
    {
      std::vector<at::Tensor> u_solved_groups;
      std::vector<at::Tensor> g_groups;
      _run_implicit_segment(seg, state, u_solved_groups, g_groups);
      _assert(static_cast<bool>(seg.ift_loader),
              "aoti::Model::jacobian: implicit segment is missing its ift_package loader. "
              "Re-export via `neml2-compile`.");
      _run_implicit_segment_jacobian(seg, u_solved_groups, g_groups, dstate);
    }
  }

  // Pack outputs and the assembled flat J of shape (*B, sum(out), sum(in)).
  std::map<std::string, at::Tensor> outputs;
  for (const auto & n : _output_names)
    outputs[n] = state.at(n);

  std::vector<at::Tensor> out_blocks;
  out_blocks.reserve(_output_names.size());
  for (const auto & n : _output_names)
  {
    auto it = dstate.find(n);
    _assert(it != dstate.end(),
            "aoti::Model::jacobian: dstate missing entry for master output '",
            n,
            "'.");
    out_blocks.push_back(it->second);
  }
  auto J = at::cat(out_blocks, /*dim=*/-2).contiguous();

  return {std::move(outputs), std::move(J)};
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
Model::Impl::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  auto [outputs, J] = _jacobian_flat(inputs); // J: (*B, Σout, Σin)

  // The batch shape is everything in J ahead of the trailing (Σout, Σin) axes.
  std::vector<int64_t> batch(J.sizes().begin(), J.sizes().end() - 2);

  // Slice the flat J into unflattened variable-pair blocks: row range
  // [out_offset, +out_var) x col range [in_offset, +in_var), reshaped to
  // (*B, *out_base, *in_base).
  VariablePairJacobian jac;
  int64_t row = 0;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
  {
    const auto out_sz = _output_sizes[i];
    auto & row_map = jac[_output_names[i]];
    for (std::size_t j = 0; j < _input_names.size(); ++j)
    {
      const auto in_sz = _input_sizes[j];
      auto block = J.narrow(-2, row, out_sz).narrow(-1, _input_offsets[j], in_sz);
      std::vector<int64_t> block_shape = batch;
      block_shape.insert(
          block_shape.end(), _output_base_shapes[i].begin(), _output_base_shapes[i].end());
      block_shape.insert(
          block_shape.end(), _input_base_shapes[j].begin(), _input_base_shapes[j].end());
      row_map[_input_names[j]] = block.reshape(block_shape).contiguous();
    }
    row += out_sz;
  }

  return {std::move(outputs), std::move(jac)};
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::jvp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & tangents) const
{
  // Compose against the internal flat Jacobian: J @ v where v is the canonical
  // input tangent stacked in declared order. One well-tested path drives both
  // jacobian() and jvp(); only the final pack differs (here we unflatten the
  // result to each output's base shape).
  auto [outputs, J] = _jacobian_flat(inputs); // J: (*B, Σout, Σin)
  std::vector<int64_t> batch(J.sizes().begin(), J.sizes().end() - 2);

  // Pack the tangent vector v: shape (*B, Σin). Each tangent is canonical
  // (*B, *in_base) and flattens to its var_size slot; a missing tangent is zero.
  std::vector<at::Tensor> v_parts;
  v_parts.reserve(_input_names.size());
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    const auto sz = _input_sizes[k];
    auto it = tangents.find(n);
    at::Tensor part;
    if (it != tangents.end())
    {
      _validate_input_shape(k, it->second); // same canonical (*B, *base) contract
      auto shape = _batch_shape_of(k, it->second);
      shape.push_back(sz); // flatten trailing base axes into the var_size slot
      part = it->second.reshape(shape).to(_dtype);
    }
    else
    {
      std::vector<int64_t> shape = batch;
      shape.push_back(sz);
      part = at::zeros(shape, J.options());
    }
    v_parts.push_back(part.contiguous());
  }
  auto v = at::cat(v_parts, /*dim=*/-1).unsqueeze(-1); // (*B, Σin, 1)
  auto Jv = at::matmul(J, v).squeeze(-1);              // (*B, Σout)

  // Split Jv per output and unflatten to the output's natural (*B, *out_base).
  std::map<std::string, at::Tensor> jvp_outputs;
  int64_t offset = 0;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
  {
    const auto sz = _output_sizes[i];
    auto flat = Jv.narrow(-1, offset, sz); // (*B, out_var_size)
    std::vector<int64_t> shape = batch;
    shape.insert(shape.end(), _output_base_shapes[i].begin(), _output_base_shapes[i].end());
    jvp_outputs[_output_names[i]] = flat.reshape(shape).contiguous();
    offset += sz;
  }

  return {std::move(outputs), std::move(jvp_outputs)};
}

} // namespace neml2::aoti
