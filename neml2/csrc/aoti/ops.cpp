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
std::map<std::string, at::Tensor>
Model::Impl::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  std::map<std::string, at::Tensor> state;
  for (const auto & n : _input_names)
  {
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::forward: missing required input '", n, "'.");
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
Model::Impl::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  // Pack state from caller inputs.
  std::map<std::string, at::Tensor> state;
  for (const auto & n : _input_names)
  {
    auto it = inputs.find(n);
    _assert(it != inputs.end(), "aoti::Model::jacobian: missing required input '", n, "'.");
    state[n] = it->second.contiguous();
  }

  // dstate[var] is (*B, var_size, M) where M = _input_total_size. Initialized
  // with identity columns for each master input, then composed segment-by-
  // segment in lockstep with `state`.
  const auto & ref = state.at(_input_names.front());
  // Infer batch shape from a master input. We assume the input arrives as
  // (*B, *base_shape) where base_shape's product == its `var_size`. The
  // simplest heuristic: the trailing axes match `var_size`, so split off the
  // leading axes as batch. Match the size exactly to disambiguate.
  const auto in_sz = _input_sizes.front();
  std::vector<int64_t> batch_shape_vec;
  {
    int64_t trail = 1;
    int64_t cut = ref.dim();
    while (cut > 0 && trail < in_sz)
    {
      --cut;
      trail *= ref.size(cut);
    }
    _assert(trail == in_sz,
            "aoti::Model::jacobian: could not isolate batch shape from input '",
            _input_names.front(),
            "' (var_size=",
            in_sz,
            ", tensor sizes=",
            ref.sizes(),
            ").");
    for (int64_t d = 0; d < cut; ++d)
      batch_shape_vec.push_back(ref.size(d));
  }
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

  // Pack outputs and assembled J of shape (*B, sum(out), sum(in)).
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

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
Model::Impl::jvp(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & tangents) const
{
  // Compose against the full Jacobian. For the typical use case (single
  // batched tangent vector) this is the right answer: J @ v where v is the
  // input tangent stacked in declared order. Implementing this on top of
  // jacobian() trades some performance for a single, well-tested code path.
  auto [outputs, J] = jacobian(inputs);

  // Pack the tangent vector v: shape (*B, sum(in_sizes)). Missing tangents
  // default to zero.
  std::vector<at::Tensor> v_parts;
  v_parts.reserve(_input_names.size());
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto & n = _input_names[k];
    auto it = tangents.find(n);
    at::Tensor part;
    if (it != tangents.end())
    {
      // Flatten trailing base axes into a single var_size slot.
      const auto sz = _input_sizes[k];
      auto t = it->second;
      int64_t trail = 1;
      int64_t cut = t.dim();
      while (cut > 0 && trail < sz)
      {
        --cut;
        trail *= t.size(cut);
      }
      _assert(trail == sz,
              "aoti::Model::jvp: tangent for input '",
              n,
              "' has shape ",
              t.sizes(),
              " that does not collapse to var_size=",
              sz);
      std::vector<int64_t> new_shape;
      for (int64_t d = 0; d < cut; ++d)
        new_shape.push_back(t.size(d));
      new_shape.push_back(sz);
      part = t.reshape(new_shape).to(_dtype);
    }
    else
    {
      // Zero tangent; piggyback on J's leading batch shape (and trailing in dim).
      const auto sz = _input_sizes[k];
      std::vector<int64_t> shape(J.sizes().begin(), J.sizes().end() - 2);
      shape.push_back(sz);
      part = at::zeros(shape, J.options());
    }
    v_parts.push_back(part.contiguous());
  }
  auto v = at::cat(v_parts, /*dim=*/-1).unsqueeze(-1); // (*B, sum(in), 1)
  auto Jv = at::matmul(J, v).squeeze(-1);              // (*B, sum(out))

  // Split Jv back per output (kept flat-trailing; caller can reshape via
  // output_sizes() if needed).
  std::map<std::string, at::Tensor> jvp_outputs;
  int64_t offset = 0;
  for (std::size_t i = 0; i < _output_names.size(); ++i)
  {
    const auto sz = _output_sizes[i];
    jvp_outputs[_output_names[i]] = Jv.narrow(-1, offset, sz).contiguous();
    offset += sz;
  }

  return {std::move(outputs), std::move(jvp_outputs)};
}

} // namespace neml2::aoti
