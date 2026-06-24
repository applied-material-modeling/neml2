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
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
/// Tunables for the implicit-segment Newton solve. Supplied at load time
/// (schema v4+ no longer bakes these into the artifact) -- the stub `.i`'s
/// `[Solvers]` block is parsed by the loader and forwarded via
/// `Model::set_solver_config`. Line search is enabled iff `ls_max_iters > 1`;
/// `ls_type` is "BACKTRACKING" or "STRONG_WOLFE".
struct SolverConfig
{
  double atol = 1.0e-10;
  double rtol = 1.0e-8;
  std::size_t miters = 25;
  std::string ls_type = "BACKTRACKING";
  std::size_t ls_max_iters = 1;
  double ls_cutback = 2.0;
  double ls_c = 1.0e-3;
};

/// A variable-pair Jacobian: `J[out_name][in_name]` is the unflattened block
/// `(*B, *out_base_shape, *in_base_shape)` (e.g. SR2->SR2 -> `(*B,6,6)`;
/// Scalar->SR2 -> `(*B,6)`; R2->R2 -> `(*B,3,3,3,3)`). Outer keys are the
/// `output_names()`, inner keys the `input_names()`. A constant (out, in) pair
/// is present with an all-zero block.
using VariablePairJacobian = std::map<std::string, std::map<std::string, at::Tensor>>;

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
 * **PImpl + visibility.** Every implementation detail -- the per-segment
 * graph state, the Newton/IFT machinery, the metadata layout -- lives behind
 * an opaque `Impl` (see the internal `internal.h`, which is not shipped). The
 * installed header carries only the public surface, and the shared library
 * (`-fvisibility=hidden` + the generated `AOTI_EXPORT` macro) exports only the
 * symbols below. Downstream consumers compile against a stable ABI that does
 * not change when the internals are reorganised.
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
 * See `doc/content/model_compilation/aoti_packages.md` for the current
 * schema metadata spec (kept in sync with the loader via
 * `kSupportedSchemaVersion`).
 */
class AOTI_EXPORT Model
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
  ///
  /// `device_override` refines *which* concrete device the artifact is loaded
  /// onto. Its device *type* must match the compiled type recorded in the
  /// metadata (a `cpu` artifact cannot be loaded onto cuda, or vice versa);
  /// only the index is honoured. This is how one `cuda` artifact is
  /// instantiated on `cuda:0`, `cuda:1`, … by a multi-device dispatcher. When
  /// omitted, the metadata device (default cuda index) is used unchanged.
  explicit Model(const std::filesystem::path & meta_path,
                 std::optional<at::Device> device_override = std::nullopt);

  /// Declared (not defaulted) here and defined out-of-line where `Impl` is a
  /// complete type, as the PImpl idiom requires.
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

  /// Per-name base shape (e.g. Scalar -> `{}`, SR2 -> `{6}`, R2 -> `{3,3}`).
  /// Inputs must be passed at their canonical `(*B, *base_shape)` shape; the
  /// flat per-variable size is `prod(base_shape)`.
  const std::vector<std::vector<int64_t>> & input_base_shapes() const noexcept;
  const std::vector<std::vector<int64_t>> & output_base_shapes() const noexcept;

  /// Evaluate the model. `inputs` is keyed by the names returned by
  /// `input_names()` and shaped `(*B, *base_shape)`; missing keys throw and a
  /// non-canonical trailing shape is rejected. Returns one tensor per name in
  /// `output_names()` at `(*B, *out_base_shape)`.
  ///
  /// `param_overrides` (default empty) replaces a promoted parameter's value for
  /// this call only, without mutating `named_parameters()`. It exists for the
  /// multi-device dispatcher, which passes each chunk its batched parameters
  /// sliced to the chunk's rows so the per-device stored params stay immutable
  /// (and concurrent workers never race). Each override is taken at the same
  /// `(*B, *param_base)` contract a stored parameter would be.
  std::map<std::string, at::Tensor>
  forward(const std::map<std::string, at::Tensor> & inputs,
          const std::map<std::string, at::Tensor> & param_overrides = {}) const;

  /// Evaluate + JVP. `tangents` shares its keys + `(*B, *in_base)` shapes with
  /// `inputs`; a missing key defaults to zero. Returns `{outputs, jvp_outputs}`
  /// -- both maps keyed by `output_names()`; `jvp_outputs[name]` is the
  /// directional derivative at the output's natural `(*B, *out_base_shape)`.
  /// See `forward` for `param_overrides`.
  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents,
      const std::map<std::string, at::Tensor> & param_overrides = {}) const;

  /// Evaluate + full Jacobian as unflattened variable-pair blocks. Returns
  /// `{outputs, J}` where `J[out_name][in_name]` is `(*B, *out_base, *in_base)`
  /// (see @ref VariablePairJacobian). Composed across forward segments and
  /// threaded through IFT blocks for implicit segments. See `forward` for
  /// `param_overrides`.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs,
           const std::map<std::string, at::Tensor> & param_overrides = {}) const;

  /// Evaluate + parameter Jacobian (schema v7). Returns `{outputs, P}` where
  /// `P[out_name][param_qname]` is the dense block `(*B, *out_base, *param_base)`
  /// -- the parameter analogue of `jacobian()` (reverse-mode AD over the promoted
  /// parameters). Requires the artifact was compiled with `-d OUT:PARAM` over a
  /// promoted parameter; otherwise raises. Supported for single-forward-segment
  /// artifacts (multi-segment / implicit parameter Jacobians are a follow-up).
  /// See `forward` for `param_overrides`.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  param_jacobian(const std::map<std::string, at::Tensor> & inputs,
                 const std::map<std::string, at::Tensor> & param_overrides = {}) const;

  /// Parameter VJP / adjoint (schema v7). Returns `dL/d(param)` keyed by
  /// parameter qualified name, for the loss `L = sum_o <cotangent_o, out_o>`.
  /// `cotangents` is keyed by output name (each at the output's `(*B, *out_base)`
  /// shape); the cheaper form for many-parameter inverse optimization. Same
  /// compile requirement + single-forward-segment support as `param_jacobian`.
  /// See `forward` for `param_overrides`. (The adjoint graph keeps parameters
  /// scalar, so per-element BATCHED parameters are not yet supported here.)
  std::map<std::string, at::Tensor>
  param_vjp(const std::map<std::string, at::Tensor> & inputs,
            const std::map<std::string, at::Tensor> & cotangents,
            const std::map<std::string, at::Tensor> & param_overrides = {}) const;

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

  /// Natural base shape of each promoted parameter (Scalar -> `{}`, SR2 ->
  /// `{6}`), keyed by qualified name. A stored parameter is `(*pbatch, *base)`;
  /// this is the trailing `*base`. The multi-device dispatcher uses it to tell a
  /// batched parameter (`pbatch == B`) from an unbatched one when slicing per
  /// chunk. Empty when nothing was promoted.
  const std::map<std::string, std::vector<int64_t>> & parameter_base_shapes() const noexcept;

  /// Replace a promoted parameter's value (the runtime-flexible setter). *name*
  /// must be a promoted parameter (appear in `named_parameters()`), else throws.
  /// The new value is taken at the parameter's `(*param_base)` shape; it is used
  /// on the next `forward`/`jvp`/`jacobian`/`param_*` call.
  void set_parameter(const std::string & name, const at::Tensor & value);

  /// Device + dtype the model was compiled for. Read-only -- baked at export
  /// time, immutable for the life of this object.
  at::Device device() const noexcept;
  at::ScalarType dtype() const noexcept;

  /// Configure the implicit-segment Newton solve (convergence tolerances,
  /// iteration cap, line search). Schema v4+ no longer bakes these into the
  /// artifact; the loader parses the stub's `[Solvers]` block and forwards it
  /// here. If never called, sensible defaults apply (see `SolverConfig`).
  void set_solver_config(const SolverConfig & config);

private:
  // Opaque implementation. Defined in the internal (non-shipped) internal.h
  // and the aoti translation units; never visible to consumers of this header.
  //
  // AOTI_NO_EXPORT overrides the type visibility the enclosing AOTI_EXPORT
  // class would otherwise propagate onto this nested type -- without it, every
  // Impl member would land in the shared library's export table.
  struct AOTI_NO_EXPORT Impl;
  std::unique_ptr<Impl> _impl;
};
} // namespace neml2::aoti
