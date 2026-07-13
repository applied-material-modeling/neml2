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
#include "neml2/csrc/aoti/krylov.h"

namespace torch::inductor
{
class AOTIModelPackageLoader;
}

namespace neml2::aoti
{
// The abstract residual/step provider the shared Newton loop drives; concrete
// backends (AOTINonlinearSystem / KrylovAOTINonlinearSystem) are built by
// `_make_implicit_system`. Defined in nonlinear_system.h.
class NonlinearSystem;

/// Lazily register NEML2's custom Torch operators (currently `neml2::opaque_pow`)
/// into the dispatcher -- once per process, only if absent. Called from the aoti
/// `Model` constructor rather than a static initializer so a process that also
/// embeds the Python neml2 package (cpp-eager / py-aoti) does not double-`def` the
/// op (the Python package registers the same schema). Defined in `custom_ops.cpp`.
void ensure_neml2_custom_ops_registered();

/// Opaque implementation of `Model`. Holds the per-segment AOTI graph state and
/// all the value / Jacobian machinery; the public `Model` methods are one-line
/// forwarders onto the identically-named members here.
struct Model::Impl
{
  /// See `Model::Model`. Parses `<artifact_root>/metadata.json`, loads every
  /// `.pt2` segment from the `<device>/<dtype>/` leaf, and materialises the
  /// promoted-parameter surface on `device` (floating params at `dtype`).
  explicit Impl(const std::filesystem::path & artifact_root,
                at::Device device,
                at::ScalarType dtype);

  // --- Public ops (forwarded from Model) -----------------------------------
  //
  // `param_overrides` (default empty) substitutes a promoted parameter's value
  // for the duration of THIS call only, without mutating the stored
  // `_named_parameters`. The multi-device dispatcher uses it to pass each chunk
  // its batched parameters sliced to the chunk's rows: the per-device Model's
  // stored params stay immutable, so concurrent workers never race. Empty
  // overrides => the stored parameters are used (the direct, single-call case).
  std::map<std::string, at::Tensor>
  forward(const std::map<std::string, at::Tensor> & inputs,
          const std::map<std::string, at::Tensor> & param_overrides = {}) const;
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents,
      const std::map<std::string, at::Tensor> & param_overrides = {}) const;
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & param_overrides = {}) const;
  /// Evaluate + parameter Jacobian. `P[out][param]` is
  /// `(*B, *out_base, *param_base)` (reverse-mode AD over promoted parameters).
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  param_jacobian(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & param_overrides = {}) const;
  /// Parameter VJP / adjoint: `dL/d(param)` for
  /// `L = sum_o <cotangent_o, out_o>`, keyed by parameter qualified name.
  std::map<std::string, at::Tensor>
  param_vjp(const std::map<std::string, at::Tensor> & inputs,
            const std::map<std::string, at::Tensor> & cotangents,
            const std::map<std::string, at::Tensor> & param_overrides = {}) const;

  // --- Accessors (forwarded from Model) ------------------------------------
  //
  // Boundary renames (shallow): with an active alias map the public surface
  // reports the RENAMED (boundary) names, while every internal structure below
  // keeps the ORIGINAL authored names. `input_names` / `output_names` /
  // `parameter_base_shapes` therefore return the pre-built ext views when
  // `_has_aliases`; `named_parameters` is itself stored boundary-keyed (see
  // `_named_parameters`), so it needs no view. Base-shape vectors are positional
  // (aligned with the name vectors) so they are returned as-is either way.
  const std::vector<std::string> & input_names() const noexcept
  {
    return _has_aliases ? _ext_input_names : _input_names;
  }
  const std::vector<std::string> & output_names() const noexcept
  {
    return _has_aliases ? _ext_output_names : _output_names;
  }
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
  /// Natural base shape per promoted parameter (Scalar => {}, SR2 => {6}). The
  /// dispatcher needs this to tell a batched stored parameter `(*pbatch, *base)`
  /// from an unbatched one when slicing per chunk. Boundary-keyed on the public
  /// surface (via the ext view) so it lines up with `named_parameters()`.
  const std::map<std::string, std::vector<int64_t>> & parameter_base_shapes() const noexcept
  {
    return _has_aliases ? _ext_param_base_shapes : _param_base_shapes;
  }

  /// The boundary (renamed) name for an ORIGINAL promoted-parameter name --
  /// identity when the parameter is unaliased. `_named_parameters` and the
  /// per-call `_param_overrides` are stored/keyed by boundary name, so every
  /// internal read (all through `_resolve_param`, plus the ctor's existence
  /// check) maps its original segment/metadata name through this first.
  const std::string & _param_boundary_name(const std::string & orig) const noexcept
  {
    auto it = _param_orig2ext.find(orig);
    return it == _param_orig2ext.end() ? orig : it->second;
  }
  at::Device device() const noexcept { return _device; }
  at::ScalarType dtype() const noexcept { return _dtype; }

  /// Resolve a promoted parameter for the CURRENT call: the per-call override
  /// value if one was supplied for `name`, else the stored `_named_parameters`
  /// entry. The single point every value/derivative graph reads a promoted
  /// parameter through, so a dispatcher-supplied per-chunk slice transparently
  /// replaces the stored value without mutating it. Throws if `name` is neither.
  const at::Tensor & _resolve_param(const std::string & name) const;

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

    /// Adaptive substepping depth cap (implicit segments only).
    /// 0 = off (single shot). >0 = the host-side driver may recursively bisect a
    /// failing increment down to this depth, interpolating / chaining / scaling
    /// the givens by their per-variable ``role``.
    int max_substepping_level = 0;

    // Forward-segment-only.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> fwd_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> jvp_loader;
    std::vector<std::string> fwd_inputs;
    std::vector<std::string> fwd_outputs;

    // Implicit-segment-only. The residual Jacobian is emitted as an OPERATOR
    // (`jacobian_loader`: (*u,*g,*p) -> (*A_blocks, *b_groups)); the linear solve
    // is a separate graph (`solve_loader`: (*A_blocks, *b_groups) -> (*du_groups))
    // that bakes the configured linear solver. `AOTINonlinearSystem::step()`
    // chains jacobian -> solve. `residual_loader` is the cheap residual eval for
    // line-search trials.
    //
    // Which loaders are present depends on the solver kind (top-level
    // `solver_config.solver_kind`): a DIRECT solve has jacobian + solve; a KRYLOV
    // (matrix-free) solve has `matvec_loader` (J.v) instead of `solve_loader`, and
    // `jacobian_loader` only when a preconditioner or an input derivative needs
    // the assembled A. Each loader is therefore constructed iff its package key is
    // present in the segment metadata.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> residual_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> jacobian_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> solve_loader;
    /// Matrix-free residual jvp J.v = ∂r/∂u . v (Krylov solvers only; null for a
    /// direct solve). `KrylovAOTINonlinearSystem::step()` drives it per inner
    /// Krylov iteration in place of chaining jacobian -> solve.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> matvec_loader;
    /// Authored preconditioner graphs (Krylov + a preconditioner only; both null
    /// for no preconditioner). `precond_setup_loader`: (*u,*g,*p) -> (*state);
    /// `precond_apply_loader`: (*state, r_flat) -> z_flat. The C++ holds the state
    /// between setup and applies, rebuilding per the cache strategy.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> precond_setup_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> precond_apply_loader;
    /// IFT (input-derivative) operator + solve graphs. `jacobian_given_loader`
    /// emits B = ∂r/∂g; A = ∂r/∂u is reused from `jacobian_loader`. `solve_ift_loader`
    /// takes `(*A_blocks, *B_blocks)` and emits one `-du/dg` block per (unknown,
    /// given) pair in `jacobian_pairs` order. Both null unless an input derivative
    /// was compiled.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> jacobian_given_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> solve_ift_loader;
    /// Implicit-segment parameter sensitivity graphs: du/dθ for a promoted
    /// parameter inside the residual. `dr_dparam_loader` emits the dense A = ∂r/∂u
    /// and ∂r/∂θ (reverse-mode AD; parameters enter PER-BATCH -- the runtime
    /// broadcasts the stored scalar to the batch before the call); `solve_param_loader`
    /// takes `(A_dense, dr_dparam)` and emits one dense du/dθ block per (unknown,
    /// param) pair in `param_jacobian_pairs` order. Both null unless an implicit
    /// parameter derivative was compiled.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> dr_dparam_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> solve_param_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> predictor_loader;

    /// Sensitivity (derivative) linear-solve kind per site (schema v12). "direct"
    /// runs the compiled `solve_ift_loader` / `solve_param_loader` graph; "krylov"
    /// runs the shared C++ Krylov loop (`krylov_solve_dense`) over the assembled A
    /// (matvec = A.v) using `input_sensitivity_krylov` / `param_sensitivity_krylov`.
    /// A krylov site's `solve_*` loader is absent -- the operators still come from
    /// `jacobian` + `jacobian_given` (IFT) or `dr_dparam` (ParamIFT). Independent of
    /// the FORWARD solve kind (`_solver_kind`): a direct forward can have an
    /// iterative sensitivity solve and vice versa.
    std::string input_sensitivity_kind = "direct";
    std::string param_sensitivity_kind = "direct";
    KrylovConfig input_sensitivity_krylov;
    KrylovConfig param_sensitivity_krylov;

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
      // Substep role: how the host-side substep driver treats this
      // given across sub-steps -- "old_state" (chain the solved unknown),
      // "old_force" / "cur_force" (interpolate the paired force), "incremental"
      // (scale by the span fraction), "static" / "unknown" (hold). ``pair`` is
      // the counterpart variable name (the paired force, or the base unknown for
      // old_state); empty when none. Only populated on an implicit segment's
      // ``givens``; the defaults leave single-shot behavior unchanged.
      std::string role = "static";
      std::string pair;
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
      /// Row/col offset of this (out_var, in_var) block within the flat
      /// `-A^{-1}B` sensitivity matrix (row = unknown storage, col = given
      /// storage). Used only by the iterative (`input_sensitivity_kind ==
      /// "krylov"`) path to slice the Krylov solution; the direct `solve_ift`
      /// loader disassembles internally. Sizes are `prod(out_base_shape)` /
      /// `prod(in_base_shape)` (plain-batch / dense).
      int64_t row_offset = 0;
      int64_t col_offset = 0;
    };
    std::vector<PairInfo> jacobian_pairs;

    /// Per-(out_var, param) parameter-Jacobian-pair metadata. Empty
    /// unless parameter derivatives were compiled. The param-Jacobian loader
    /// returns one dense block per pair in this row-major order (outputs outer,
    /// promoted params inner) and ONLY blocks (no value outputs).
    struct ParamPairInfo
    {
      std::string out_var;
      std::string param;
      std::vector<int64_t> out_base_shape;
      std::vector<int64_t> param_base_shape;
      /// Row/col offset of this (out_var, param) block within the flat
      /// `-A^{-1} ∂r/∂θ` matrix (row = unknown storage, col = ∂r/∂θ param
      /// storage). Used only by the iterative (`param_sensitivity_kind ==
      /// "krylov"`) path to slice the Krylov solution; the direct `solve_param`
      /// loader disassembles internally.
      int64_t row_offset = 0;
      int64_t col_offset = 0;
    };
    std::vector<ParamPairInfo> param_jacobian_pairs;
    /// Forward-segment parameter-Jacobian graph. Its promoted-parameter inputs
    /// are PER-BATCH (`(*B, *param_base)`) -- the runtime broadcasts the stored
    /// scalar parameter to the batch before the call (the value / jvp graphs keep
    /// parameters scalar). Null unless parameter derivatives were compiled.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> param_jacobian_loader;

    /// Parameter VJP / adjoint graph: inputs are the model inputs
    /// (parameters scalar) followed by one TYPED cotangent per output (in
    /// `param_vjp_outputs` order); outputs are the parameter gradients (in
    /// `param_vjp_params` order). Null unless parameter derivatives were compiled.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> param_vjp_loader;
    std::vector<std::string> param_vjp_params;
    std::vector<std::string> param_vjp_outputs;

    std::vector<std::string> predictor_inputs;
    std::vector<std::string> predictor_outputs;
    /// Promoted parameters the predictor graph consumes. Separate
    /// from `param_inputs` because the predictor is a distinct nn.Module: the
    /// the operator graphs take the residual's promoted tail, but the predictor
    /// is compiled without it, so it currently receives no promoted params
    /// (empty). Kept as its own field so threading promoted params into a
    /// predictor later does not perturb the residual tail.
    std::vector<std::string> predictor_param_inputs;

    // Solver convergence / line-search configuration is no longer per-segment
    // metadata; it lives in the owning Impl's `_solver_config`,
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

  /// Broadcast a stored promoted parameter `(*pbatch, *base)` to the runtime call
  /// batch `(*batch, *base)` (right-aligned; `base` is its natural base, the
  /// trailing `base_ndim` dims). A scalar/unbatched parameter expands up to the
  /// batch; a per-batch-element parameter (`pbatch == batch`) passes through. The
  /// value / jvp / jacobian / param-Jacobian graphs all take promoted parameters
  /// as per-batch inputs.
  static at::Tensor broadcast_param_to_batch(const at::Tensor & param,
                                             const std::vector<int64_t> & batch,
                                             int64_t base_ndim);

  /// Run a forward segment: pull inputs from `state`, run AOTI (with the
  /// promoted-parameter tail, each parameter broadcast to `batch`), write outputs
  /// back to `state`.
  void _run_forward_segment(const Segment & seg,
                            std::map<std::string, at::Tensor> & state,
                            const std::vector<int64_t> & batch) const;

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

  /// Build the `NonlinearSystem` the shared C++ Newton loop drives for an
  /// implicit segment: an `AOTINonlinearSystem` (direct: jacobian -> solve) or a
  /// `KrylovAOTINonlinearSystem` (matrix-free Krylov over the matvec graph),
  /// chosen by `_solver_kind`. `g_groups` are the per-group givens (bound into
  /// the system). Shared by the plain + masked implicit solve paths.
  std::unique_ptr<NonlinearSystem>
  _make_implicit_system(const Segment & seg, const std::vector<at::Tensor> & g_groups) const;

  /// Adaptive substepping driver: wraps `_run_implicit_segment` in a
  /// recursive bisection over the increment. Snapshots the segment's endpoint
  /// givens, then solves the increment in one shot; on a recoverable
  /// ConvergenceError it halves the span -- interpolating each paired force,
  /// scaling each listed increment, and chaining each old-state to the previous
  /// sub-step's solved unknown, per the given's `role` -- recursing up to
  /// `seg.max_substepping_level` levels before re-raising. On return `state`
  /// holds the final converged unknowns and the out-params hold the LAST
  /// sub-step's per-group solved unknowns + givens (the point the IFT path,
  /// when substepping the Jacobian, evaluates at). Only called when
  /// `seg.max_substepping_level > 0`.
  void _run_implicit_segment_substepped(const Segment & seg,
                                        std::map<std::string, at::Tensor> & state,
                                        std::vector<at::Tensor> & u_solved_groups,
                                        std::vector<at::Tensor> & g_groups) const;

  /// Masked single implicit solve: like `_run_implicit_segment` but drives
  /// `Newton::solve_masked` and returns the per-element convergence mask (bool,
  /// dynamic-batch shape) WITHOUT throwing. Converged rows' solved unknowns are
  /// written to `state`; the mask tells the substep driver which rows to freeze
  /// vs bisect. `_masking_ok` gates whether masking applies (1-D dynamic batch).
  at::Tensor _run_implicit_segment_masked(const Segment & seg,
                                          std::map<std::string, at::Tensor> & state) const;

  /// Per-element (masked) substepping: solve only the still-unconverged subset of
  /// the dynamic batch at each sub-step, freezing converged rows at their
  /// coarsest converging solution and bisecting only the failing subset. The
  /// value + Jacobian variants mirror `_run_implicit_segment_substepped[_jacobian]`
  /// but scatter converged rows into full-batch accumulators. Require a 1-D
  /// dynamic batch (see `_masking_ok`); `ops.cpp` falls back to the whole-batch
  /// driver otherwise. Only called when `seg.max_substepping_level > 0`.
  void _run_implicit_segment_substepped_masked(const Segment & seg,
                                               std::map<std::string, at::Tensor> & state) const;
  void _run_implicit_segment_substepped_masked_jacobian(
      const Segment & seg,
      std::map<std::string, at::Tensor> & state,
      std::map<std::string, at::Tensor> & dstate) const;

  /// True iff masking applies to `state` for this segment: the dynamic batch is
  /// a single axis (dim 0), so per-row `index_select`/`index_copy` cleanly
  /// selects elements. Multi-axis dynamic batches fall back to whole-batch.
  bool _masking_ok(const Segment & seg, const std::map<std::string, at::Tensor> & state) const;

  /// Substepping analogue of `_run_implicit_segment_jacobian`: solves the
  /// increment by the same bisection AND accumulates the chained consistent
  /// tangent into `dstate`. At each successfully-solved leaf sub-span it runs
  /// the value solve then the IFT composition, having first written the span's
  /// givens into `dstate` per their role -- interpolated force sensitivities,
  /// scaled increments, and each old-state seeded with the previous span's
  /// solved-unknown sensitivity -- so one IFT call yields
  /// `J_k = A_k·J_{k-1} + B_k·frac_k·J_endpoint`. On return `state` holds the
  /// final unknowns and `dstate[unknown]` the total `du_M/d(master inputs)`.
  /// Requires `seg.solve_ift_loader`. Only called when `seg.max_substepping_level > 0`.
  void _run_implicit_segment_substepped_jacobian(const Segment & seg,
                                                 std::map<std::string, at::Tensor> & state,
                                                 std::map<std::string, at::Tensor> & dstate) const;

  /// Write the substep sub-span [a, b] transform of a name->tensor map into
  /// `dst`, reading the endpoint snapshot `orig` and the chained old-state
  /// `chained`. The coefficient math is identical for the primal state and the
  /// dstate (chain-rule) carrier -- interpolate a paired force, scale a listed
  /// increment, take the chained value for an old-state, hold anything else --
  /// so both the value and Jacobian drivers share it.
  void _apply_substep_span(const Segment & seg,
                           const std::map<std::string, at::Tensor> & orig,
                           double a,
                           double b,
                           const std::map<std::string, at::Tensor> & chained,
                           std::map<std::string, at::Tensor> & dst) const;

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
                                     std::map<std::string, at::Tensor> & dstate,
                                     const std::vector<int64_t> & batch) const;

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

  /// The DYNAMIC (plain) batch of a canonical input: the leading axes before
  /// both the per-input sub-batch axes (`_input_sub_batch_shapes[idx]`) and the
  /// trailing `base_shape`. Unlike `_batch_shape_of` (which returns dynamic +
  /// sub-batch), this excludes the sub-batch axes -- the structural per-site
  /// axes (e.g. crystal-plasticity per-grain / per-slip) that must NOT be
  /// broadcast across inputs. Used by `_prepare_inputs` to unify only the plain
  /// batch across a mix of global and sub-batched inputs.
  std::vector<int64_t> _dynamic_batch_shape_of(std::size_t idx, const at::Tensor & t) const;

  /// Validate every required input and return them as a `state` map with the
  /// dynamic-batch axes broadcast to a single common shape (base axes
  /// preserved), so a batch-independent input -- e.g. MOOSE's scalar TIME force
  /// -- is lifted to the call batch instead of colliding with the batched
  /// inputs in a downstream `cat`. This is the C++ analogue of
  /// `neml2/types/_boundary.py::broadcast_to_common_batch`, which the typed
  /// (eager / py-aoti shim) routes already apply at their raw-tensor boundary.
  std::map<std::string, at::Tensor>
  _prepare_inputs(const std::map<std::string, at::Tensor> & inputs) const;

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

  // Master (out, param) parameter-derivative pairs the artifact supports (schema
  // v7), in metadata order. Empty => no parameter-Jacobian graph; param_jacobian
  // / param_vjp raise.
  std::vector<std::pair<std::string, std::string>> _param_derivatives;

  // Narrowed parameter-column layout for the multi-segment parameter-Jacobian
  // carrier: the distinct requested promoted parameters (in `_param_derivatives`
  // order), with their column offsets / sizes in the `(*B, var_folded, P)`
  // parameter carrier (`P == _param_total_size`). Parallel to `_req_input_*`.
  std::vector<std::string> _req_params;
  std::map<std::string, int64_t> _req_param_offset;
  std::map<std::string, int64_t> _req_param_size;
  int64_t _param_total_size = 0;

  /// Single-forward-segment parameter-Jacobian fast path: broadcast the stored
  /// scalar parameters to the runtime batch, run the param-Jacobian loader once,
  /// and return (outputs, per-(out,param) blocks). Only valid for one forward
  /// segment carrying a param-Jacobian loader.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  _forward_param_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const;

  /// Single-implicit-segment parameter-Jacobian path: run the Newton
  /// solve to convergence, then run the ParamIFT loader on the converged
  /// per-group unknowns + givens + the promoted parameters broadcast per-batch,
  /// returning (outputs == converged unknowns, per-(unknown, param) du/dθ
  /// blocks). Only valid for one implicit segment carrying a param-IFT loader.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  _implicit_param_pair_blocks(const std::map<std::string, at::Tensor> & inputs) const;

  /// Multi-segment (composed) parameter-Jacobian carrier. Mirrors
  /// `_jacobian_dstate` but for promoted parameters: a `(*B, var_folded, P)`
  /// carrier seeded to ZERO (master inputs do not depend on parameters), then
  /// composed segment by segment. Each forward segment adds its INDIRECT
  /// contribution (`jvp_loader` per-pair blocks composed against `dpstate[in]`
  /// via `_run_forward_segment_jacobian`) plus its DIRECT contribution
  /// (`param_jacobian_loader` blocks injected at the parameter's column band);
  /// each implicit segment adds its indirect (the IFT operator+solve graphs via
  /// `_run_implicit_segment_jacobian`) plus its direct (`solve_param_loader`
  /// blocks). Returns the master outputs + the per-variable `dpstate` map; the
  /// caller slices each requested `(out, param)` block.
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  _param_jacobian_dstate(const std::map<std::string, at::Tensor> & inputs) const;

  /// Add a forward segment's DIRECT parameter Jacobian into `dpstate`: run the
  /// segment's `param_jacobian_loader` (promoted params broadcast per-batch) and
  /// accumulate each `d(out)/d(param)` block at the parameter's column band of
  /// `dpstate[out]`. `common_dyn` is the carrier's shared dynamic batch.
  void _add_forward_param_direct(const Segment & seg,
                                 const std::map<std::string, at::Tensor> & state,
                                 const std::vector<int64_t> & common_dyn,
                                 std::map<std::string, at::Tensor> & dpstate) const;

  /// Add an implicit segment's DIRECT parameter sensitivity into `dpstate`: run
  /// the segment's dr_dparam + solve_param graphs on the converged per-group unknowns +
  /// givens + promoted params (broadcast per-batch) and accumulate each
  /// `du/d(param)` block at the parameter's column band of `dpstate[unknown]`.
  void _add_implicit_param_direct(const Segment & seg,
                                  const std::vector<at::Tensor> & u_solved_groups,
                                  const std::vector<at::Tensor> & g_groups,
                                  const std::vector<int64_t> & common_dyn,
                                  std::map<std::string, at::Tensor> & dpstate) const;

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

  // Per-call promoted-parameter override (null outside a call). Points at the
  // caller's `param_overrides` map for the duration of one public op, consulted
  // by `_resolve_param`. `mutable` because the public ops are const yet need to
  // stash it; thread-safe under the dispatcher because each per-device Model is
  // driven by a single worker at a time (the stored params are never mutated).
  // Set/restored by `ParamOverrideGuard` so nested internal calls (e.g.
  // param_vjp -> param_jacobian) inherit the outer override.
  mutable const std::map<std::string, at::Tensor> * _param_overrides = nullptr;

  /// RAII setter for `_param_overrides`. A non-empty `overrides` installs itself
  /// for the guard's lifetime; an empty one leaves the current override in place
  /// (so an internal call that passes no override inherits its caller's). The
  /// previous pointer is always restored on destruction.
  struct ParamOverrideGuard
  {
    const Impl * impl;
    const std::map<std::string, at::Tensor> * prev;
    ParamOverrideGuard(const Impl * i, const std::map<std::string, at::Tensor> & overrides)
      : impl(i),
        prev(i->_param_overrides)
    {
      if (!overrides.empty())
        i->_param_overrides = &overrides;
    }
    ~ParamOverrideGuard() { impl->_param_overrides = prev; }
    ParamOverrideGuard(const ParamOverrideGuard &) = delete;
    ParamOverrideGuard & operator=(const ParamOverrideGuard &) = delete;
  };

  // Natural base shape per promoted parameter, from its typed class
  // (Scalar => {}, SR2 => {6}). Used to split a (possibly batched) stored
  // parameter `(*pbatch, *base)` into batch vs base -- for broadcasting it to the
  // call batch and for reshaping parameter-derivative blocks. Distinct from the
  // stored tensor's full shape, which carries any batch dim. Keyed by ORIGINAL
  // name (every internal reader uses original names); the boundary-keyed view is
  // `_ext_param_base_shapes`.
  std::map<std::string, std::vector<int64_t>> _param_base_shapes;

  // --- Boundary renames (shallow) ------------------------------------------
  // Optional `boundary_aliases` from the metadata. Only the names reported at
  // the public surface change; every internal structure above keeps the ORIGINAL
  // authored names. `_*_orig2ext` / `_*_ext2orig` are the per-namespace forward /
  // reverse maps (present only for renamed entries); `_ext_*` are the pre-built
  // boundary views the public accessors return. `_named_parameters` and the
  // per-call `_param_overrides` are themselves stored boundary-keyed, so params
  // need only the forward map (`_param_orig2ext`, consulted by
  // `_param_boundary_name`). `_has_aliases` gates the facade's key translation so
  // the fully-baked / unrenamed common case pays nothing.
  bool _has_aliases = false;
  std::map<std::string, std::string> _in_orig2ext, _in_ext2orig;
  std::map<std::string, std::string> _out_orig2ext, _out_ext2orig;
  std::map<std::string, std::string> _param_orig2ext;
  std::vector<std::string> _ext_input_names, _ext_output_names;
  std::map<std::string, std::vector<int64_t>> _ext_param_base_shapes;

  // Device + dtype baked into the artifact at export time.
  at::Device _device = at::kCPU;
  at::ScalarType _dtype = at::kDouble;

  // Implicit-segment Newton configuration, supplied at load time via
  // Model::set_solver_config (defaults if never set).
  SolverConfig _solver_config;

  // Implicit linear-solver kind (baked at compile, read from metadata):
  // "direct" chains jacobian -> solve; "krylov" runs a matrix-free Krylov solve
  // (`_krylov_config`) over `matvec_loader`. `_run_implicit_segment` branches on
  // this to pick the AOTI / Krylov NonlinearSystem.
  std::string _solver_kind = "direct";
  KrylovConfig _krylov_config;
};
} // namespace neml2::aoti
