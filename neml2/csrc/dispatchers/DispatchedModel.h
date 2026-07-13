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

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/dispatchers/WorkScheduler.h"
#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
/**
 * @brief A `Model`-shaped wrapper that dispatches a batched workload behind a
 *        scheduler.
 *
 * Owns the pre-loaded per-device `Model`(s) produced by `neml2-compile` and an
 * injected @ref WorkScheduler, and re-exposes the exact `forward` / `jvp` /
 * `jacobian` surface of @ref Model. Each call slices the input batch along its
 * leading axis into scheduler-sized chunks, moves each chunk to its compute
 * device, runs that device's `Model`, moves the result back to the device the
 * inputs were provided on, and concatenates -- all transparently.
 *
 * Sync vs. async is chosen from the scheduler's type:
 * - a @ref SyncScheduler (`SimpleScheduler`, `MPISimpleScheduler`) runs the
 *   chunk loop on the calling thread, one device. When that device equals the
 *   input device and the batch fits in one chunk, it short-circuits to a direct
 *   `Model` call (zero overhead).
 * - an @ref AsyncScheduler (`StaticHybridScheduler`) drives a thread-per-device
 *   pool: the calling thread asks the scheduler for the next `(device, chunk)`
 *   and enqueues it; one worker per device runs its `Model` concurrently;
 *   results are reassembled in dispatch order.
 *
 * This is a *distinct, same-shaped* type, **not** a subclass of `Model` (whose
 * methods are non-virtual by design): substitute it for `Model` at the source
 * level. The scheduler is held by composition so the policy stays swappable.
 *
 * Artifact layout. `artifact_root` is the directory `neml2-compile` writes: one
 * shared `metadata.json` at the root plus per-`<device>/<dtype>/` `.pt2` binaries
 * One `Model` is loaded per `scheduler->devices()` entry from the
 * `<device-type>/<dtype>/` leaf, pinned to that device's concrete index; the
 * shared metadata (structural + solver config) backs them all.
 *
 * Multi-device semantics. `input_names()` / `output_names()` / `*_sizes()` /
 * `dtype()` are identical across the per-device copies and forward to the
 * primary (`scheduler->devices().front()`); `device()` reports that primary.
 * `named_parameters()` is the single *master* promoted-parameter map: mutating
 * it in place is broadcast to every device copy before the next dispatch (so the
 * single-device mutation idiom keeps working under hybrid).
 */
class AOTI_EXPORT DispatchedModel
{
public:
  /// Load one per-device artifact under `artifact_root` for each
  /// `scheduler->devices()` entry (from the `<device>/<dtype>/` leaf), and
  /// dispatch through `scheduler`. Throws if a device's `<device>/<dtype>/` leaf
  /// or the shared `metadata.json` is missing.
  DispatchedModel(const std::filesystem::path & artifact_root,
                  std::shared_ptr<WorkScheduler> scheduler,
                  at::ScalarType dtype = at::kDouble);

  /// Wrap an already-loaded `Model`. The scheduler's device *type* must match
  /// the model's compiled device type. Convenience for callers that hold a
  /// `Model` directly rather than an artifact-root directory.
  DispatchedModel(std::unique_ptr<Model> model, std::shared_ptr<WorkScheduler> scheduler);

  /// Defined out-of-line where `Impl` is complete (PImpl idiom).
  ~DispatchedModel();

  // Non-copyable. Movable: the handle owns its state through `unique_ptr<Impl>`
  // (the inner Models are reached via that pointer), so moving the handle is
  // sound even though `Model` itself is non-movable. Move lets factories return
  // a DispatchedModel by value. Declared here, defaulted out-of-line where
  // `Impl` is complete.
  DispatchedModel(const DispatchedModel &) = delete;
  DispatchedModel & operator=(const DispatchedModel &) = delete;
  DispatchedModel(DispatchedModel &&) noexcept;
  DispatchedModel & operator=(DispatchedModel &&) noexcept;

  /// @name Model surface (chunked + dispatched). Mirrors Model exactly.
  ///@{
  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const;

  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const;

  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  /// Evaluate + dense parameter Jacobian, chunked + dispatched. Returns
  /// `{outputs, P}` with `P[out_name][param_qname]` at `(*B, *out_base,
  /// *param_base)`; blocks are concatenated across chunks (batch-independent
  /// blocks passed through), mirroring `jacobian()`.
  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  param_jacobian(const std::map<std::string, at::Tensor> & inputs) const;

  /// Parameter VJP / adjoint `dL/d(param)`, chunked + dispatched. Each chunk
  /// returns the adjoint summed over its batch slice, so the per-parameter grads
  /// are SUMMED across chunks (not concatenated).
  std::map<std::string, at::Tensor>
  param_vjp(const std::map<std::string, at::Tensor> & inputs,
            const std::map<std::string, at::Tensor> & cotangents) const;
  ///@}

  /// Configure the implicit-segment Newton solve (forwarded to every Model).
  void set_solver_config(const SolverConfig & config);

  /// @name Metadata + parameter surface.
  /// Metadata forwards to the primary device copy (all copies agree);
  /// named_parameters() is the master map broadcast to all copies per dispatch.
  ///@{
  const std::vector<std::string> & input_names() const noexcept;
  const std::vector<std::string> & output_names() const noexcept;
  const std::vector<std::vector<int64_t>> & input_base_shapes() const noexcept;
  const std::vector<std::vector<int64_t>> & output_base_shapes() const noexcept;
  /// Per-promoted-parameter natural base shape, keyed by qualified name (the
  /// unified parameter surface; forwards to the primary device copy, all agree).
  const std::map<std::string, std::vector<int64_t>> & parameter_base_shapes() const noexcept;
  std::map<std::string, at::Tensor> & named_parameters() noexcept;
  const std::map<std::string, at::Tensor> & named_parameters() const noexcept;

  /// Replace a promoted parameter's value on the master copy; re-synced to every
  /// device copy on the next dispatched call. Mirrors `Model::set_parameter`.
  void set_parameter(const std::string & name, const at::Tensor & value);
  /// The compute device the workload is dispatched to.
  at::Device device() const noexcept;
  at::ScalarType dtype() const noexcept;
  ///@}

private:
  struct AOTI_NO_EXPORT Impl;
  std::unique_ptr<Impl> _impl;
};
} // namespace neml2::aoti
