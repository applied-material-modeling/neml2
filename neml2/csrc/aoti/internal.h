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
  /// materialises the promoted-parameter surface. `device_override` refines the
  /// concrete device index (its type must match the compiled device type).
  explicit Impl(const std::filesystem::path & meta_path, std::optional<at::Device> device_override);

  // --- Public ops (forwarded from Model) -----------------------------------
  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const;
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const;
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  // --- Accessors (forwarded from Model) ------------------------------------
  const std::vector<std::string> & input_names() const noexcept { return _input_names; }
  const std::vector<std::string> & output_names() const noexcept { return _output_names; }
  const std::vector<std::vector<int64_t>> & input_base_shapes() const noexcept
  {
    return _input_base_shapes;
  }
  const std::vector<std::vector<int64_t>> & output_base_shapes() const noexcept
  {
    return _output_base_shapes;
  }
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
      /// True when this block does not depend on the dynamic batch (e.g. a
      /// constant stiffness tensor). The compiled graph emits it with a static
      /// size-1 leading dynamic-batch axis; the single-forward-segment Jacobian
      /// path squeezes that axis and returns the block unbatched.
      bool batch_independent = false;
    };
    std::vector<PairInfo> jacobian_pairs;

    std::vector<std::string> predictor_inputs;
    std::vector<std::string> predictor_outputs;

    // Solver convergence / line-search configuration is no longer per-segment
    // metadata (schema v4); it lives in the owning Impl's `_solver_config`,
    // set at load time from the stub's [Solvers] block.

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
  /// receive one block per ``(unknown, given)`` pair of ``-du_dg`` (in
  /// ``seg.jacobian_pairs`` order, the disassemble of the IFT matrix), and
  /// accumulate each block's matmul against ``dstate[given]`` into the
  /// ``dstate[unknown]`` slot -- the same per-pair Jacobian composition a
  /// forward segment uses. (BLOCK / per-grain unknowns are rejected with a
  /// clear "not yet implemented" error pending the sub-batch IFT path.)
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

  /// Reject a non-canonical input: the tensor's trailing axes must equal the
  /// declared `base_shape` for `name` (so a Scalar is `(*B,)`, an SR2 `(*B,6)`).
  /// Throws FatalError otherwise. `idx` is the index of `name` in `_input_names`.
  void _validate_input_shape(std::size_t idx, const at::Tensor & t) const;

  /// Split a canonical input tensor into its leading batch shape (everything
  /// before the trailing `base_shape` axes). Assumes `_validate_input_shape`
  /// has already passed for this input.
  std::vector<int64_t> _batch_shape_of(std::size_t idx, const at::Tensor & t) const;

  /// Compose the master Jacobian carrier: returns the output values plus the
  /// per-variable `dstate` map, where `dstate[var]` is `(*common_dyn,
  /// var_folded, M_req)` -- the variable's sensitivity to the requested input
  /// columns (`M_req`), its own sub-batch folded into `var_folded`. `jacobian()`
  /// slices each requested output's block and reshapes to `(*B, *out_base,
  /// *in_base)`; `jvp()` contracts the requested input columns with the
  /// tangents. (Returning the map -- not a single catted `(*B, Σout, M_req)` --
  /// keeps per-output offsets correct when outputs have heterogeneous folded
  /// sizes, e.g. a per-grain output alongside a global one.)
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  _jacobian_dstate(const std::map<std::string, at::Tensor> & inputs) const;

  // The per-group Newton iteration + line-search reductions now live in
  // newton.{h,cpp} (driven through the NonlinearSystem abstraction);
  // `_run_implicit_segment` builds an AOTINonlinearSystem and calls Newton.

  // Per-segment runtime state, in declared order.
  std::vector<Segment> _segments;

  // Master IO (in graph-call order). The public surface is the per-variable
  // base shapes; the flat _input_sizes / _output_sizes / _input_offsets /
  // _input_total_size (== prod(base_shape) and its prefix sums) stay private,
  // driving the internal flat dstate / J assembly math.
  std::vector<std::string> _input_names;
  std::vector<std::vector<int64_t>> _input_base_shapes;
  std::vector<int> _input_sizes;
  std::vector<int64_t> _input_offsets;
  int64_t _input_total_size = 0;
  std::vector<std::string> _output_names;
  std::vector<std::vector<int64_t>> _output_base_shapes;
  std::vector<int> _output_sizes;

  // Per-master-input sub-batch shapes (empty when plain-batch). The dense
  // Jacobian carrier seeds `dstate[var] = (*common_dyn, var_folded, M_req)` --
  // a shared dynamic batch (NOT the first input's full leading shape) with each
  // variable's own sub-batch axes folded into the storage dim -- so a
  // heterogeneous mix of global and per-grain inputs composes correctly.
  std::vector<std::vector<int64_t>> _input_sub_batch_shapes;

  // Master (out, in) derivative pairs the artifact supports, in the order the
  // metadata recorded them (output-order, input-order). Empty => no derivative
  // graphs were compiled; jvp() / jacobian() raise. `_deriv_by_out` is a cache
  // mapping each covered output to its requested inputs for O(1) assembly.
  std::vector<std::pair<std::string, std::string>> _derivatives;
  std::map<std::string, std::vector<std::string>> _deriv_by_out;

  // The "dense auxiliary B matrix" narrowing (multi-segment path): the dense
  // chain-rule carrier (`dstate`) and flat Jacobian carry columns for only the
  // *requested* input directions, not every master input. `_req_inputs` is the
  // distinct requested inputs in `_input_names` order; `_req_input_offset` /
  // `_req_total_size` are their column offsets / total width in that narrowed
  // matrix.
  std::vector<std::string> _req_inputs;
  std::map<std::string, int64_t> _req_input_offset;
  std::map<std::string, int64_t> _req_input_size;
  int64_t _req_total_size = 0;

  /// Single-forward-segment Jacobian fast path: run the segment's jvp loader once
  /// and return (outputs, per-pair blocks) directly (no dense flat-J), squeezing
  /// the static size-1 dynamic-batch axis of batch-independent blocks so they
  /// come back unbatched. Only valid when there is exactly one Forward segment
  /// with a jvp loader.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  _forward_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const;

  /// True iff the artifact is a single forward segment carrying a jvp loader --
  /// the case the per-pair fast path (unbatched batch-independent return) covers.
  bool _is_single_forward_jac() const
  {
    return _segments.size() == 1 && _segments.front().kind == SegmentKind::Forward &&
           static_cast<bool>(_segments.front().jvp_loader);
  }

  // Owned state -- only the runtime-flexible (promoted) parameters live here.
  // Baked entries are constants inside the AOTI graphs and have no C++-side
  // representation.
  std::map<std::string, at::Tensor> _named_parameters;

  // Device + dtype baked into the artifact at export time.
  at::Device _device = at::kCPU;
  at::ScalarType _dtype = at::kDouble;

  // Implicit-segment Newton configuration, supplied at load time via
  // Model::set_solver_config (schema v4+; defaults if never set).
  SolverConfig _solver_config;
};
} // namespace neml2::aoti
