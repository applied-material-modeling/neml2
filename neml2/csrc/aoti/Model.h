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

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

namespace torch::inductor
{
class AOTIModelPackageLoader;
}

namespace neml2::aoti
{
/**
 * @brief Thin, self-contained runtime for AOTI-exported NEML2 models.
 *
 * Loads the `_meta.json` + per-segment `.pt2` artifacts produced by
 * `neml2-compile` and exposes them as a plain C++ object: three public
 * operations (`forward` / `jvp` / `jacobian`) keyed by string names, and an
 * owned, mutable parameter surface (`named_parameters()`) for the entries the
 * user explicitly promoted at compile time via `neml2-compile --parameter`.
 *
 * **No NEML2Object inheritance.** This class does not participate in the
 * Registry / Factory / OptionSet machinery. It is constructed directly from
 * a filesystem path -- the loader (Python pybind binding, future driver
 * code) is responsible for parsing any HIT input via `nmhit` and extracting
 * the `meta` field to pass here. This decoupling is deliberate: the C++→
 * will eventually retire the C++ Factory/Registry surface
 * entirely, and this class is shaped to outlive it.
 *
 * Bake-by-default contract
 * ------------------------
 * Every parameter and buffer in the source model is folded into the AOTI
 * graph as a constant unless the user promoted it via `--parameter NAME`.
 * Baked entries are immutable post-compile; promoted entries are graph inputs
 * and live as live C++ state, mutable through `named_parameters()`. A model
 * compiled with no `--parameter` flags has an empty `named_parameters()` and
 * is effectively a frozen inference artifact.
 *
 * Device pinning
 * --------------
 * The AOTI graphs are pinned to a specific device + dtype at export time and
 * cannot be migrated after load. Promoted-parameter tensors are placed on the
 * same device. There is intentionally no `to()` -- offering one would either
 * be a misleading no-op (graph stays put) or a contract-breaking half-move
 * (params shift, graph doesn't). To retarget, re-run `neml2-compile`.
 *
 * See `python/neml2/native/AOTI_PACKAGES.md` for the schema_version=2
 * metadata spec.
 */
class Model
{
public:
  /// Load all .pt2 segments + metadata from `meta_path`. Other artifact files
  /// (`.pt2` segments, optional `_state.pt`) are resolved relative to the
  /// metadata file's directory. Throws on any missing file or schema
  /// mismatch.
  ///
  /// Promoted-parameter tensors are placed on the device + dtype recorded in
  /// the metadata (set by `neml2-compile --device/--dtype`). The AOTI graphs
  /// are pinned to that same device at export time.
  explicit Model(const std::filesystem::path & meta_path);

  ~Model();

  // Non-copyable, non-movable. Held by callers as a unique_ptr / shared_ptr
  // (see pybind binding or test fixtures) or as an automatic on the stack.
  Model(const Model &) = delete;
  Model(Model &&) = delete;
  Model & operator=(const Model &) = delete;
  Model & operator=(Model &&) = delete;

  /// Master inputs / outputs in graph-call order, as recorded in the metadata.
  const std::vector<std::string> & input_names() const noexcept;
  const std::vector<std::string> & output_names() const noexcept;

  /// Per-name flat sizes (product of declared base shape; 1 for Scalar).
  const std::vector<int> & input_sizes() const noexcept;
  const std::vector<int> & output_sizes() const noexcept;

  /// Evaluate the model. `inputs` is keyed by the names returned by
  /// `input_names()`; missing keys throw. Returns one tensor per name in
  /// `output_names()`.
  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const;

  /// Evaluate + JVP. `tangents` shares its keys with `inputs`; missing keys
  /// default to zero. Returns `{outputs, jvp_outputs}` -- both maps keyed by
  /// `output_names()`.
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const;

  /// Evaluate + full Jacobian. The returned `J` is the assembled
  /// `(*B, sum(out_sizes), sum(in_sizes))` block-stacked tensor that the
  /// segments' `_jvp.pt2` graphs already produce; the wrapper composes across
  /// forward segments and threads IFT blocks across implicit segments. Per-
  /// output / per-input slabs can be sliced off using `output_sizes()` /
  /// `input_sizes()` offsets.
  std::pair<std::map<std::string, at::Tensor>, at::Tensor>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  /// Mutable surface for the runtime-flexible parameters (the set promoted
  /// via `neml2-compile --parameter NAME`). Baked entries do not appear here.
  /// Empty when the model was compiled with no promotions.
  ///
  /// The thin class owns the underlying tensors; mutating an entry in-place
  /// is reflected on the next `forward`/`jvp`/`jacobian` call. Replacing the
  /// tensor via assignment is allowed too -- the entry's dtype and shape
  /// must match the compile-time contract, or the next forward will throw.
  std::map<std::string, at::Tensor> & named_parameters() noexcept;
  const std::map<std::string, at::Tensor> & named_parameters() const noexcept;

  /// Device + dtype the model was compiled for. Read-only -- baked at export
  /// time, immutable for the life of this object.
  at::Device device() const noexcept { return _device; }
  at::ScalarType dtype() const noexcept { return _dtype; }

private:
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
    std::vector<int> fwd_input_sizes;
    std::vector<std::string> fwd_outputs;
    std::vector<int> fwd_output_sizes;

    // Implicit-segment-only.
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> rhs_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> step_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> ift_loader;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> predictor_loader;
    std::vector<std::string> givens;
    std::vector<int> given_sizes;
    std::vector<std::vector<int64_t>> given_unflattened_shapes;
    std::vector<std::string> unknowns;
    std::vector<int> unknown_sizes;
    std::vector<std::vector<int64_t>> unknown_unflattened_shapes;
    std::vector<std::string> predictor_inputs;
    std::vector<std::string> predictor_outputs;
    int64_t u_size = 0;
    int64_t g_size = 0;
    double atol = 0.0;
    double rtol = 0.0;
    std::size_t miters = 0;

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

  /// Run an implicit segment: pull givens (and optional predictor inputs)
  /// from `state`, run the predictor (if loaded) + Newton loop, write the
  /// converged unknowns to `state`. Out-params hand the converged `u_solved`
  /// and packed `g_flat` to the caller so the IFT loader can run on the same
  /// tensors without re-packing.
  void _run_implicit_segment(const Segment & seg,
                             std::map<std::string, at::Tensor> & state,
                             at::Tensor & u_solved,
                             at::Tensor & g_flat) const;

  /// Run a forward segment's JVP loader and compose its Jacobian into
  /// `dstate`. Replaces the value-loader call: the JVP loader returns
  /// `(*outputs, J)`, so outputs land in `state` and `J` is matmul'd against
  /// the row-stacked `dstate[seg.fwd_inputs]` and split into per-output
  /// blocks written to `dstate[seg.fwd_outputs[i]]`.
  void _run_forward_segment_jacobian(const Segment & seg,
                                     std::map<std::string, at::Tensor> & state,
                                     std::map<std::string, at::Tensor> & dstate) const;

  /// Compose an implicit segment's IFT into `dstate`: run the IFT loader on
  /// `(u_solved, g_flat)`, row-stack `dstate` blocks for `seg.givens`,
  /// batched-matmul, split into per-unknown blocks written to
  /// `dstate[seg.unknowns[i]]`. Predictor-only inputs (not in `givens`) carry
  /// structurally-zero columns automatically (Newton converges from any u0).
  void _run_implicit_segment_jacobian(const Segment & seg,
                                      const at::Tensor & u_solved,
                                      const at::Tensor & g_flat,
                                      std::map<std::string, at::Tensor> & dstate) const;

  /// Initialize `dstate` with identity columns for each master input.
  /// `dstate[_input_names[k]]` is `(*B, s_k, M)` with the `(s_k, s_k)`
  /// identity placed at columns `[c_k, c_k + s_k)` and zeros elsewhere.
  void _init_dstate(const at::TensorOptions & options,
                    at::IntArrayRef batch_shape,
                    std::map<std::string, at::Tensor> & dstate) const;

  /// Fused two-graph Newton loop. Loads `rhs` once for the baseline residual,
  /// then iterates `step` (which fuses assemble + solve + update + residual)
  /// up to `miters` times or until `||b|| < atol` or `||b||/||b0|| < rtol`.
  /// Promoted-parameter tail is threaded into both loaders.
  at::Tensor
  _solve_newton(const Segment & seg, const at::Tensor & u0, const at::Tensor & g_flat) const;

  /// Reshape `(*B, *unflattened_shape)` → `(*B, var_size)` flat slot the
  /// implicit graphs expect. Handles the Scalar edge case (`unflattened_shape`
  /// empty, `var_size == 1`) by adding a trailing length-1 axis.
  static at::Tensor _reshape_to_flat_slot(const at::Tensor & t,
                                          int64_t var_size,
                                          const std::vector<int64_t> & unflattened_shape);

  /// Inverse of `_reshape_to_flat_slot`: `(*B, var_size)` →
  /// `(*B, *unflattened_shape)`. Drops the size-1 trailing axis for Scalar.
  static at::Tensor _reshape_from_flat_slot(const at::Tensor & slice,
                                            int64_t var_size,
                                            const std::vector<int64_t> & unflattened_shape);

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
