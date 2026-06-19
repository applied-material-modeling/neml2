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
#include "neml2/csrc/aoti/WorkScheduler.h"
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
 * leading axis into scheduler-sized chunks, moves each chunk to the compute
 * device, runs the device's `Model`, moves the result back to the device the
 * inputs were provided on, and concatenates -- all transparently. When the
 * scheduler's device equals the input device and the batch fits in one chunk,
 * the behaviour is identical to calling `Model` directly.
 *
 * This is a *distinct, same-shaped* type, **not** a subclass of `Model` (whose
 * methods are non-virtual by design) and not behind an abstract interface:
 * substitute it for `Model` at the source level, e.g. via a template. The
 * scheduler is held by composition so the policy stays swappable.
 *
 * Artifact layout. `artifact_root` is the directory `neml2-compile --device
 * <...>` writes, holding one subfolder per device named by device type
 * (`cpu/`, `cuda/`), each with a complete `*_meta.json` + `.pt2` segments. The
 * subfolder matching the scheduler's device type is loaded, pinned to the
 * scheduler's concrete device index.
 *
 * Single-device scope. In this synchronous phase a scheduler maps to exactly
 * one device, so the wrapper holds one `Model`; `device()` / `dtype()` /
 * `named_parameters()` forward to it. Multi-device storage and the matching
 * multi-valued semantics are a later (asynchronous / hybrid) concern.
 */
class AOTI_EXPORT DispatchedModel
{
public:
  /// Load the per-device artifact under `artifact_root` matching `scheduler`'s
  /// device, and dispatch through `scheduler`. Throws if the subfolder or its
  /// metadata is missing.
  DispatchedModel(const std::filesystem::path & artifact_root,
                  std::shared_ptr<WorkScheduler> scheduler);

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

  std::pair<std::map<std::string, at::Tensor>, at::Tensor>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const;
  ///@}

  /// Configure the implicit-segment Newton solve (forwarded to every Model).
  void set_solver_config(const SolverConfig & config);

  /// @name Metadata + parameter forwarders.
  /// Valid while single-device (this phase); multi-valued once >1 Model is held.
  ///@{
  const std::vector<std::string> & input_names() const noexcept;
  const std::vector<std::string> & output_names() const noexcept;
  const std::vector<int> & input_sizes() const noexcept;
  const std::vector<int> & output_sizes() const noexcept;
  std::map<std::string, at::Tensor> & named_parameters() noexcept;
  const std::map<std::string, at::Tensor> & named_parameters() const noexcept;
  /// The compute device the workload is dispatched to.
  at::Device device() const noexcept;
  at::ScalarType dtype() const noexcept;
  ///@}

private:
  struct AOTI_NO_EXPORT Impl;
  std::unique_ptr<Impl> _impl;
};
} // namespace neml2::aoti
