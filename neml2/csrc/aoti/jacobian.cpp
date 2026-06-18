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
  std::vector<int64_t> shape_vec(batch_shape.begin(), batch_shape.end());
  for (std::size_t k = 0; k < _input_names.size(); ++k)
  {
    const auto s_k = _input_sizes[k];
    const auto c_k = _input_offsets[k];
    shape_vec.push_back(s_k);
    shape_vec.push_back(_input_total_size);
    auto block = at::zeros(shape_vec, options);
    block.narrow(/*dim=*/-1, /*start=*/c_k, /*length=*/s_k).copy_(at::eye(s_k, options));
    shape_vec.pop_back();
    shape_vec.pop_back();
    dstate[_input_names[k]] = block;
  }
}

void
Model::Impl::_run_forward_segment_jacobian(const Segment & seg,
                                           std::map<std::string, at::Tensor> & state,
                                           std::map<std::string, at::Tensor> & dstate) const
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
  for (auto & p : _gather_params(seg.param_inputs))
    inputs.push_back(std::move(p));

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
  // IFT loader signature: (*u_groups, *g_groups, [params...]) -> *cells.
  // Each cell is one (row_group, col_group) entry of -du_dg in row-major
  // order (matches seg.ift_cells).
  std::vector<at::Tensor> ift_in;
  ift_in.reserve(u_solved_groups.size() + g_groups.size() + seg.param_inputs.size());
  for (const auto & t : u_solved_groups)
    ift_in.push_back(t.contiguous());
  for (const auto & t : g_groups)
    ift_in.push_back(t.contiguous());
  for (auto & p : _gather_params(seg.param_inputs))
    ift_in.push_back(std::move(p));

  const auto ift_outs = seg.ift_loader->run(ift_in);
  _assert(ift_outs.size() == seg.ift_cells.size(),
          "aoti::Model: IFT loader returned ",
          ift_outs.size(),
          " tensors, expected ",
          seg.ift_cells.size(),
          " (one per cell).");

  // Master_ndim from dstate of any given var (all dstate entries share
  // the trailing M dim).
  _assert(!seg.givens.empty(),
          "aoti::Model: implicit-segment IFT has no givens; cannot infer master_ndim.");
  auto first_dg_it = dstate.find(seg.givens[0].name);
  _assert(first_dg_it != dstate.end(),
          "aoti::Model: dstate missing given '",
          seg.givens[0].name,
          "' at IFT composition time.");
  const int64_t M = first_dg_it->second.size(-1);

  // Batch shape (leading dims of any cell's tensor, before its sub +
  // base axes). Pick cell 0 to source it.
  const auto & cell0 = ift_outs[0];
  const auto & cinfo0 = seg.ift_cells[0];
  const int64_t cell0_trail =
      static_cast<int64_t>(cinfo0.sub_batch_shape.size()) + 2; // sub + 2 base
  _assert(cell0.dim() >= cell0_trail,
          "aoti::Model: IFT cell tensor ndim=",
          cell0.dim(),
          " < expected trail=",
          cell0_trail);
  const int64_t batch_ndim = cell0.dim() - cell0_trail;
  std::vector<int64_t> batch_shape;
  batch_shape.reserve(static_cast<std::size_t>(batch_ndim));
  for (int64_t d = 0; d < batch_ndim; ++d)
    batch_shape.push_back(cell0.size(d));

  // Per-unknown accumulator at the converged-segment shape:
  // (*B, u.var_size, M). Mirrors what the old flat-matmul produced so
  // downstream forward composition sees an identical layout.
  std::map<std::string, at::Tensor> du_acc;
  for (const auto & u : seg.unknowns)
  {
    std::vector<int64_t> shape = batch_shape;
    shape.push_back(u.var_size);
    shape.push_back(M);
    du_acc[u.name] = at::zeros(shape, cell0.options());
  }

  // Iterate cells; for each, slice per-(rvar, cvar) sub-cells on the
  // last 2 dims by base offsets, dispatch matmul kind by
  // (row_structure, col_structure), accumulate into du_acc[rvar].
  for (std::size_t k = 0; k < ift_outs.size(); ++k)
  {
    const auto & cell_t = ift_outs[k];
    const auto & cinfo = seg.ift_cells[k];
    const bool row_block = (cinfo.row_structure == "block");
    const bool col_block = (cinfo.col_structure == "block");

    int64_t r_off = 0;
    for (const auto & rv : cinfo.row_vars)
    {
      int64_t r_base = 1;
      for (auto s : rv.base_shape)
        r_base *= s;
      int64_t r_sub_total = 1;
      for (auto s : rv.sub_batch_shape)
        r_sub_total *= s;

      int64_t c_off = 0;
      for (const auto & cv : cinfo.col_vars)
      {
        int64_t c_base = 1;
        for (auto s : cv.base_shape)
          c_base *= s;
        int64_t c_sub_total = 1;
        for (auto s : cv.sub_batch_shape)
          c_sub_total *= s;

        // Slice sub-cell from cell_t (last-2 dims).
        auto sub_cell = cell_t.narrow(/*dim=*/-2, /*start=*/r_off, /*length=*/r_base)
                            .narrow(/*dim=*/-1, /*start=*/c_off, /*length=*/c_base);

        // Pick the matching dg_dmaster view for cv. dstate[cv.name]
        // has shape (*B, cv.var_size, M). For a BLOCK col group, view
        // as (*B, *cv.sub, *cv.base_flat, M) but we want (*B, sub, c_base, M)
        // with sub_total flat across cv's sub axes; for DENSE col, keep
        // (*B, cv.var_size, M).
        auto dg_it = dstate.find(cv.name);
        _assert(dg_it != dstate.end(),
                "aoti::Model: IFT consumer needs dstate['",
                cv.name,
                "'] which is missing.");
        const auto & dg = dg_it->second;

        // Contribution shape: (*B, *cell_sub_for_row, r_base, M).
        at::Tensor contribution;
        if (row_block && col_block)
        {
          // PAIRED BLOCK + BLOCK: sub axes on row and col are the same
          // (single sub axis on cell tensor). dg viewed as
          // (*B, c_sub_total, c_base, M).
          std::vector<int64_t> dg_target = batch_shape;
          for (auto s : cv.sub_batch_shape)
            dg_target.push_back(s);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          // sub_cell shape: (*B, *cell_sub, r_base, c_base); dg_view
          // (*B, *cv.sub, c_base, M); matmul → (*B, *sub, r_base, M).
          contribution = at::matmul(sub_cell, dg_view);
        }
        else if (row_block && !col_block)
        {
          // BLOCK row, DENSE col: sub_cell (*B, *row_sub, r_base,
          // c_folded). dg (*B, c_folded, M). Broadcast dg with
          // singleton sub axes.
          std::vector<int64_t> dg_target = batch_shape;
          for (std::size_t s = 0; s < cinfo.sub_batch_shape.size(); ++s)
            dg_target.push_back(1);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          contribution = at::matmul(sub_cell, dg_view);
        }
        else if (!row_block && col_block)
        {
          // DENSE row, BLOCK col: sub_cell (*B, *col_sub, r_folded,
          // c_base). dg (*B, *cv.sub, c_base, M). matmul → (*B,
          // *col_sub, r_folded, M); SUM over the col_sub axes (cross-
          // grain reduction into the global row).
          std::vector<int64_t> dg_target = batch_shape;
          for (auto s : cv.sub_batch_shape)
            dg_target.push_back(s);
          dg_target.push_back(c_base);
          dg_target.push_back(M);
          auto dg_view = dg.reshape(dg_target);
          auto per_col = at::matmul(sub_cell, dg_view);
          std::vector<int64_t> sum_dims;
          const int64_t col_sub_ndim = static_cast<int64_t>(cv.sub_batch_shape.size());
          for (int64_t d = per_col.dim() - 2 - col_sub_ndim; d < per_col.dim() - 2; ++d)
            sum_dims.push_back(d);
          contribution = per_col.sum(sum_dims);
        }
        else
        {
          // DENSE row, DENSE col: standard matmul.
          contribution = at::matmul(sub_cell, dg);
        }

        // Accumulate into du_acc[rv.name] at shape (*B, rv.var_size, M).
        // For BLOCK row, contribution carries row sub axes; reshape to
        // (*B, sub_total * r_base, M) so it lays out var-shaped slabs.
        at::Tensor flat_contribution;
        if (row_block)
        {
          std::vector<int64_t> target = batch_shape;
          target.push_back(r_sub_total * r_base);
          target.push_back(M);
          flat_contribution = contribution.reshape(target);
          (void)c_sub_total; // not used in this branch
        }
        else
        {
          flat_contribution = contribution;
        }

        // r_var_size = (BLOCK row: r_sub_total * r_base; DENSE row:
        // r_base). du_acc is keyed by rv.name with shape (*B,
        // rv.var_size, M).
        int64_t r_var_size = (row_block) ? (r_sub_total * r_base) : r_base;
        _assert(r_var_size == rv.var_size,
                "aoti::Model: IFT row-var size mismatch for '",
                rv.name,
                "'");
        du_acc[rv.name] = du_acc[rv.name] + flat_contribution;

        c_off += c_base;
      }
      r_off += r_base;
    }
  }

  // Write final per-unknown du into dstate. Negation: cells carry
  // -du_dg already (Python side returns -du_dg.tensors[i][j]).
  for (const auto & u : seg.unknowns)
    dstate[u.name] = du_acc[u.name].contiguous();
}

} // namespace neml2::aoti
