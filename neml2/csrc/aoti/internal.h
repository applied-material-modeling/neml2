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

#pragma once

// Internal header -- NOT shipped. Defines `Model::Impl`, the opaque
// implementation behind the public `Model` facade in `Model.h`. The four aoti
// translation units (Model.cpp, ops.cpp, solve.cpp, jacobian.cpp) include this
// to implement `Impl`'s members; nothing outside the aoti library ever sees it.
// Everything here is compiled with hidden visibility (see CMakeLists.txt).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ATen.h is the umbrella header that pulls in the full at::Tensor template
// definitions (Tensor::item<T>(), etc.). The narrower <ATen/Functions.h>
// alone leaves Tensor declared but its member-templates unparseable -- a
// regression we hit when aoti stopped inheriting the legacy misc/tensor
// PCH that used to drag the full umbrella in.
#include <ATen/ATen.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/aoti/assertions.h"

namespace torch::inductor
{
class AOTIModelPackageLoader;
}

namespace neml2::aoti
{
/// Opaque implementation of `Model`. Holds the per-segment AOTI graph state and
/// all the value / Jacobian machinery; the public `Model` methods are one-line
/// forwarders onto the identically-named members here.
struct Model::Impl
{
  /// See `Model::Model`. Parses `_meta.json`, loads every `.pt2` segment, and
  /// materialises the promoted-parameter surface.
  explicit Impl(const std::filesystem::path & meta_path);

  // --- Public ops (forwarded from Model) -----------------------------------
  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const;
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const;
  std::pair<std::map<std::string, at::Tensor>, at::Tensor>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  // --- Accessors (forwarded from Model) ------------------------------------
  const std::vector<std::string> & input_names() const noexcept { return _input_names; }
  const std::vector<std::string> & output_names() const noexcept { return _output_names; }
  const std::vector<int> & input_sizes() const noexcept { return _input_sizes; }
  const std::vector<int> & output_sizes() const noexcept { return _output_sizes; }
  std::map<std::string, at::Tensor> & named_parameters() noexcept { return _named_parameters; }
  const std::map<std::string, at::Tensor> & named_parameters() const noexcept
  {
    return _named_parameters;
  }
  at::Device device() const noexcept { return _device; }
  at::ScalarType dtype() const noexcept { return _dtype; }

  enum class SegmentKind
  {
    Forward,
    Implicit,
  };

  /// One segment of the composed graph. All fields are populated regardless
  /// of `kind`; fields irrelevant to a given kind are unused.
  struct Segment
  {
    SegmentKind kind;

    // Forward-segment-only.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> fwd_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> jvp_loader;
    std::vector<std::string> fwd_inputs;
    std::vector<std::string> fwd_outputs;

    // Implicit-segment-only.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> rhs_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> step_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> ift_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> predictor_loader;

    /// Per-variable metadata. The natural ``(*B, *sub_batch, *base)``
    /// shape is what the AOTI graph was traced with; the C++ side
    /// packs per-variable tensors into the segment loader's positional
    /// input list in this declared order. Each ``var_size = prod(sub) *
    /// prod(base)``; ``base_ndim = base_shape.size()`` and
    /// ``sub_batch_ndim = sub_batch_shape.size()`` together with the
    /// runtime tensor's ``dim()`` give the per-segment dynamic-batch
    /// ndim (used by the convergence-norm reduction).
    struct VarInfo
    {
      std::string name;
      int64_t var_size = 0;
      std::vector<int64_t> sub_batch_shape;
      std::vector<int64_t> base_shape;
    };
    std::vector<VarInfo> unknowns;
    std::vector<VarInfo> givens;
    std::vector<VarInfo> residuals;

    /// Per-group metadata for the implicit-segment per-group I/O contract
    /// (v7+). The Newton inner loop runs entirely per-group; per-variable
    /// ↔ per-group conversion happens twice per solve, at the boundaries.
    /// ``structure`` is "block" (preserve sub_batch axes) or "dense" (sub
    /// folded into base). ``per_var_info`` lists the variables in this
    /// group in their canonical order (matches the per-variable slot at
    /// the source ``unknowns``/``givens``/``residuals`` vector).
    struct GroupInfo
    {
      std::string structure;
      std::vector<int64_t> sub_batch_shape;
      std::vector<VarInfo> per_var_info;
    };
    std::vector<GroupInfo> unknown_groups;
    std::vector<GroupInfo> given_groups;
    std::vector<GroupInfo> residual_groups;

    /// Per-(row_group, col_group) cell metadata for the IFT loader's
    /// output tuple. ``ift_cells[k]`` corresponds to ``ift_outs[k]`` --
    /// row-major (rows outer, cols inner) over ``unknown_groups`` ×
    /// ``given_groups``. Each cell carries the row/col group indices,
    /// the per-side structure ("block" / "dense"), the cell's own
    /// sub_batch_shape (paired or single-side), and the per-(rvar, cvar)
    /// VarInfo lists for slicing sub-cells out of the cell tensor.
    struct CellInfo
    {
      int64_t row_group_idx = 0;
      int64_t col_group_idx = 0;
      std::string row_structure;
      std::string col_structure;
      std::vector<int64_t> sub_batch_shape;
      std::vector<VarInfo> row_vars;
      std::vector<VarInfo> col_vars;
    };
    std::vector<CellInfo> ift_cells;

    /// Per-(out_var, in_var) Jacobian-pair metadata for the
    /// forward-segment Jacobian loader output tuple. Ordered row-major
    /// (outputs outer, structural inputs inner). The C++ Jacobian
    /// consumer iterates these in step with the trailing entries of the
    /// JVP loader's output tuple (after ``fwd_outputs.size()`` value
    /// tensors).
    struct PairInfo
    {
      std::string out_var;
      std::string in_var;
      std::vector<int64_t> out_base_shape;
      std::vector<int64_t> in_base_shape;
      std::vector<int64_t> in_sub_batch_shape;
    };
    std::vector<PairInfo> jacobian_pairs;

    std::vector<std::string> predictor_inputs;
    std::vector<std::string> predictor_outputs;

    double atol = 0.0;
    double rtol = 0.0;
    std::size_t miters = 0;

    // Line-search options. ls_max_iters == 1 means "no line search" (just
    // accept the full Newton step). Anything > 1 enables backtracking with
    // the given cutback factor and Armijo-style stopping criterion.
    // ls_type values: "BACKTRACKING" or "STRONG_WOLFE".
    std::string ls_type = "BACKTRACKING";
    std::size_t ls_max_iters = 1;
    double ls_cutback = 2.0;
    double ls_c = 1.0e-3;

    /// Names of the promoted parameters this segment's graphs consume, in
    /// graph-call order. Looked up in the owning Model's `_named_parameters`
    /// at each call; empty in the common fully-baked case and the segment's
    /// loader call shape is identical to pre-v2.
    ///
    /// In this iteration this is only ever populated for forward segments --
    /// `neml2-compile` rejects `--parameter` flags targeting attributes
    /// inside an implicit segment (the Dense/Block equation-system wrappers
    /// have fixed `(u_flat, g_flat)` signatures). When that integration
    /// lands, the predictor may end up needing its own promoted tail
    /// (its module is a separate nn.Module from the system model); a
    /// `predictor_param_inputs` field can be added then.
    std::vector<std::string> param_inputs;
  };

  /// Lower a name list to a vector of tensors pulled (in order) from
  /// `_named_parameters`. Throws if any name is missing. Common helper for
  /// every loader call.
  std::vector<at::Tensor> _gather_params(const std::vector<std::string> & names) const;

  /// Run a forward segment: pull inputs from `state`, run AOTI (with the
  /// promoted-parameter tail), write outputs back to `state`.
  void _run_forward_segment(const Segment & seg, std::map<std::string, at::Tensor> & state) const;

  /// Run an implicit segment: pack per-group ``u_groups``/``g_groups``
  /// from per-variable ``state`` (one ``at::cat`` per group along the
  /// last base axis), optionally run the predictor, run the Newton loop
  /// fully per-group, unpack the converged ``u_groups`` back to
  /// per-variable ``state`` for downstream forward consumers. Out-params
  /// hand the converged per-group ``u_groups`` and per-group ``g_groups``
  /// to the caller so the IFT loader can run on the same tensors without
  /// re-packing.
  void _run_implicit_segment(const Segment & seg,
                             std::map<std::string, at::Tensor> & state,
                             std::vector<at::Tensor> & u_solved_groups,
                             std::vector<at::Tensor> & g_groups) const;

  /// Pack per-variable ``state`` entries into per-group tensors via the
  /// AssembledVector convention: BLOCK groups cat per-var contributions
  /// along the last base axis preserving sub_batch axes; DENSE groups
  /// fold each var's sub_batch into base, then cat. Used at solve start
  /// to build ``u_groups`` / ``g_groups`` once per solve.
  std::vector<at::Tensor> _pack_groups(const std::map<std::string, at::Tensor> & state,
                                       const std::vector<Segment::GroupInfo> & groups) const;

  /// Unpack per-group tensors back into per-variable ``state`` entries.
  /// Inverse of :meth:`_pack_groups`. Called once at solve end to write
  /// the converged ``u_groups`` back to ``state[u.name]`` for downstream
  /// forward composition.
  void _unpack_groups(const std::vector<at::Tensor> & group_tensors,
                      const std::vector<Segment::GroupInfo> & groups,
                      std::map<std::string, at::Tensor> & state) const;

  /// Run a forward segment's JVP loader and compose its Jacobian into
  /// `dstate`. Replaces the value-loader call: the JVP loader returns
  /// `(*outputs, J)`, so outputs land in `state` and `J` is matmul'd against
  /// the row-stacked `dstate[seg.fwd_inputs]` and split into per-output
  /// blocks written to `dstate[seg.fwd_outputs[i]]`.
  void _run_forward_segment_jacobian(const Segment & seg,
                                     std::map<std::string, at::Tensor> & state,
                                     std::map<std::string, at::Tensor> & dstate) const;

  /// Compose an implicit segment's IFT into `dstate`: run the IFT loader on
  /// the converged per-group ``u_solved_groups`` + per-group ``g_groups``,
  /// receive one tensor per ``(row_group, col_group)`` cell of
  /// ``-du_dg`` (in row-major order matching ``seg.ift_cells``), and
  /// accumulate per-cell matmul contributions into per-unknown
  /// ``du_dmaster`` slots. Each cell's matmul kind is dispatched by
  /// ``(row_structure, col_structure)``; per-grain BLOCK + BLOCK cells
  /// stay compact throughout, no N² dense materialisation.
  void _run_implicit_segment_jacobian(const Segment & seg,
                                      const std::vector<at::Tensor> & u_solved_groups,
                                      const std::vector<at::Tensor> & g_groups,
                                      std::map<std::string, at::Tensor> & dstate) const;

  /// Initialize `dstate` with identity columns for each master input.
  /// `dstate[_input_names[k]]` is `(*B, s_k, M)` with the `(s_k, s_k)`
  /// identity placed at columns `[c_k, c_k + s_k)` and zeros elsewhere.
  void _init_dstate(const at::TensorOptions & options,
                    at::IntArrayRef batch_shape,
                    std::map<std::string, at::Tensor> & dstate) const;

  /// Per-group Newton loop (v7+). Loads `rhs` once for the baseline
  /// residual, then iterates `step` (which fuses assemble + solve +
  /// per-unknown-group step direction + per-residual-group b) up to
  /// `miters` times or until convergence. Each per-residual-group `b`
  /// and per-unknown-group `du` is one tensor at natural
  /// `(*B, *group_sub, group_folded_base)` shape (BLOCK group: sub
  /// preserved; DENSE group: sub folded into base). Line-search updates
  /// `u_groups[i] = u_groups[i] + alpha * du_groups[i]` per-group; trial
  /// residuals come from the cheap `rhs` loader on the per-group trial
  /// tensors. Returns the converged per-unknown-group vector.
  std::vector<at::Tensor> _solve_newton(const Segment & seg,
                                        const std::vector<at::Tensor> & u0_groups,
                                        const std::vector<at::Tensor> & g_groups) const;

  /// Per-group b sum-of-squares reduction across all residual groups,
  /// summing over each group's sub_batch + 1 base axis (the
  /// group_folded_base axis), leaving only the dynamic-batch leading
  /// axes. Used for convergence norm and the Armijo `b · du`
  /// Strong-Wolfe scalar.
  static at::Tensor _pergroup_norm_sq(const std::vector<at::Tensor> & group_tensors,
                                      const std::vector<Segment::GroupInfo> & groups);

  /// Pairwise sum: ``sum_k sum_{non-batch}(a_groups[k] * b_groups[k])``.
  /// Both vectors must be the same length with shape-matching entries.
  /// Used for the Armijo line-search ``b · du`` criterion.
  static at::Tensor _pergroup_dot(const std::vector<at::Tensor> & a_groups,
                                  const std::vector<at::Tensor> & b_groups,
                                  const std::vector<Segment::GroupInfo> & groups);

  /// Reshape `(*B,)` alpha-per-batch to broadcast against a per-group
  /// tensor of shape `(*B, *group_sub, group_folded_base)`. Adds
  /// `group_sub.size() + 1` trailing size-1 axes.
  static at::Tensor _alpha_for_group(const at::Tensor & alpha, const Segment::GroupInfo & group);

  // Per-segment runtime state, in declared order.
  std::vector<Segment> _segments;

  // Master IO (in graph-call order).
  std::vector<std::string> _input_names;
  std::vector<int> _input_sizes;
  std::vector<int64_t> _input_offsets;
  int64_t _input_total_size = 0;
  std::vector<std::string> _output_names;
  std::vector<int> _output_sizes;

  // Owned state -- only the runtime-flexible (promoted) parameters live here.
  // Baked entries are constants inside the AOTI graphs and have no C++-side
  // representation.
  std::map<std::string, at::Tensor> _named_parameters;

  // Device + dtype baked into the artifact at export time.
  at::Device _device = at::kCPU;
  at::ScalarType _dtype = at::kDouble;
};
} // namespace neml2::aoti
