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

#include <filesystem>
#include <utility>

#include "neml2/csrc/dispatchers/DispatchedModel.h"
#include "neml2/csrc/dispatchers/batch_chunk.h"

namespace neml2::aoti
{
namespace
{
// Find the single `*_meta.json` inside a per-device artifact subfolder.
std::filesystem::path
find_meta(const std::filesystem::path & dir)
{
  _assert(std::filesystem::is_directory(dir),
          "DispatchedModel: artifact subfolder '",
          dir.string(),
          "' does not exist. Compile with `neml2-compile --device <type>` so a per-device "
          "subfolder is produced.");

  const std::string suffix = "_meta.json";
  std::filesystem::path found;
  for (const auto & entry : std::filesystem::directory_iterator(dir))
  {
    const auto p = entry.path();
    const auto fn = p.filename().string();
    if (fn.size() >= suffix.size() &&
        fn.compare(fn.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      _assert(found.empty(),
              "DispatchedModel: multiple '*_meta.json' files in '",
              dir.string(),
              "'. Expected exactly one compiled model.");
      found = p;
    }
  }
  _assert(!found.empty(),
          "DispatchedModel: no '*_meta.json' in '",
          dir.string(),
          "'. Is this a neml2-compile output directory?");
  return found;
}
} // namespace

// ----------------------------------------------------------------------------
// Impl
// ----------------------------------------------------------------------------
struct DispatchedModel::Impl
{
  Impl(const std::filesystem::path & artifact_root, std::shared_ptr<WorkScheduler> scheduler)
    : _scheduler(std::move(scheduler))
  {
    _assert(static_cast<bool>(_scheduler), "DispatchedModel: scheduler must not be null.");
    const auto dev = _scheduler->device();
    const auto subdir = artifact_root / (dev.is_cuda() ? "cuda" : "cpu");
    const auto meta = find_meta(subdir);
    // Pin the artifact to the scheduler's concrete device (index honoured for
    // cuda:N); the override's type must match the compiled type.
    auto model = std::make_unique<Model>(meta, dev);
    _active = model.get();
    _models.emplace(dev.str(), std::move(model));
  }

  Impl(std::unique_ptr<Model> model, std::shared_ptr<WorkScheduler> scheduler)
    : _scheduler(std::move(scheduler))
  {
    _assert(static_cast<bool>(_scheduler), "DispatchedModel: scheduler must not be null.");
    _assert(static_cast<bool>(model), "DispatchedModel: model must not be null.");
    _assert(model->device().type() == _scheduler->device().type(),
            "DispatchedModel: scheduler device '",
            _scheduler->device().str(),
            "' does not match the provided model's device '",
            model->device().str(),
            "'.");
    _active = model.get();
    _models.emplace(model->device().str(), std::move(model));
  }

  /// Chunk extent along dim 0: the scheduler's batch size, clamped to the whole
  /// batch (and 0 meaning "no chunking").
  int64_t chunk_extent(int64_t b) const
  {
    const auto n = _scheduler->batch_size();
    if (n == 0 || static_cast<int64_t>(n) >= b)
      return b;
    return static_cast<int64_t>(n);
  }

  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs) const
  {
    _assert(!inputs.empty(), "DispatchedModel::forward: inputs are empty.");
    const auto in_device = inputs.begin()->second.device();
    const auto compute_device = _active->device();
    const int64_t b = infer_batch_size(inputs);
    const int64_t chunk = chunk_extent(b);

    // Fast path: one chunk on the input's own device -> no slice/transfer/concat.
    // Identical to calling Model directly.
    if (chunk >= b && compute_device == in_device)
      return _active->forward(inputs);

    std::vector<std::map<std::string, at::Tensor>> results;
    results.reserve(static_cast<std::size_t>((b + chunk - 1) / chunk));
    for (int64_t start = 0; start < b; start += chunk)
    {
      const int64_t count = std::min(chunk, b - start);
      auto in_dev = to_device(slice_batch(inputs, start, count), compute_device);
      results.push_back(to_device(_active->forward(in_dev), in_device));
    }
    return cat_batch(results);
  }

  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents) const
  {
    _assert(!inputs.empty(), "DispatchedModel::jvp: inputs are empty.");
    const auto in_device = inputs.begin()->second.device();
    const auto compute_device = _active->device();
    const int64_t b = infer_batch_size(inputs);
    const int64_t chunk = chunk_extent(b);

    if (chunk >= b && compute_device == in_device)
      return _active->jvp(inputs, tangents);

    std::vector<std::map<std::string, at::Tensor>> outs;
    std::vector<std::map<std::string, at::Tensor>> jouts;
    const auto nchunks = static_cast<std::size_t>((b + chunk - 1) / chunk);
    outs.reserve(nchunks);
    jouts.reserve(nchunks);
    for (int64_t start = 0; start < b; start += chunk)
    {
      const int64_t count = std::min(chunk, b - start);
      auto in_dev = to_device(slice_batch(inputs, start, count), compute_device);
      auto tan_dev = to_device(slice_batch(tangents, start, count), compute_device);
      auto [out, jout] = _active->jvp(in_dev, tan_dev);
      outs.push_back(to_device(out, in_device));
      jouts.push_back(to_device(jout, in_device));
    }
    return {cat_batch(outs), cat_batch(jouts)};
  }

  std::pair<std::map<std::string, at::Tensor>, at::Tensor>
  jacobian(const std::map<std::string, at::Tensor> & inputs) const
  {
    _assert(!inputs.empty(), "DispatchedModel::jacobian: inputs are empty.");
    const auto in_device = inputs.begin()->second.device();
    const auto compute_device = _active->device();
    const int64_t b = infer_batch_size(inputs);
    const int64_t chunk = chunk_extent(b);

    if (chunk >= b && compute_device == in_device)
      return _active->jacobian(inputs);

    std::vector<std::map<std::string, at::Tensor>> outs;
    std::vector<at::Tensor> jparts;
    const auto nchunks = static_cast<std::size_t>((b + chunk - 1) / chunk);
    outs.reserve(nchunks);
    jparts.reserve(nchunks);
    for (int64_t start = 0; start < b; start += chunk)
    {
      const int64_t count = std::min(chunk, b - start);
      auto in_dev = to_device(slice_batch(inputs, start, count), compute_device);
      auto [out, j] = _active->jacobian(in_dev);
      outs.push_back(to_device(out, in_device));
      jparts.push_back(j.device() == in_device ? j : j.to(in_device));
    }
    return {cat_batch(outs), cat_batch_tensor(jparts)};
  }

  void set_solver_config(const SolverConfig & config)
  {
    for (auto & [name, m] : _models)
      m->set_solver_config(config);
  }

  std::shared_ptr<WorkScheduler> _scheduler;
  // device-string -> Model. Exactly one entry in this synchronous phase; the
  // map shape is forward-looking for the multi-device (async / hybrid) phase.
  std::map<std::string, std::unique_ptr<Model>> _models;
  Model * _active = nullptr;
};

// ----------------------------------------------------------------------------
// Facade
// ----------------------------------------------------------------------------
DispatchedModel::DispatchedModel(const std::filesystem::path & artifact_root,
                                 std::shared_ptr<WorkScheduler> scheduler)
  : _impl(std::make_unique<Impl>(artifact_root, std::move(scheduler)))
{
}

DispatchedModel::DispatchedModel(std::unique_ptr<Model> model,
                                 std::shared_ptr<WorkScheduler> scheduler)
  : _impl(std::make_unique<Impl>(std::move(model), std::move(scheduler)))
{
}

DispatchedModel::~DispatchedModel() = default;
DispatchedModel::DispatchedModel(DispatchedModel &&) noexcept = default;
DispatchedModel & DispatchedModel::operator=(DispatchedModel &&) noexcept = default;

std::map<std::string, at::Tensor>
DispatchedModel::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  return _impl->forward(inputs);
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
DispatchedModel::jvp(const std::map<std::string, at::Tensor> & inputs,
                     const std::map<std::string, at::Tensor> & tangents) const
{
  return _impl->jvp(inputs, tangents);
}

std::pair<std::map<std::string, at::Tensor>, at::Tensor>
DispatchedModel::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  return _impl->jacobian(inputs);
}

void
DispatchedModel::set_solver_config(const SolverConfig & config)
{
  _impl->set_solver_config(config);
}

const std::vector<std::string> &
DispatchedModel::input_names() const noexcept
{
  return _impl->_active->input_names();
}

const std::vector<std::string> &
DispatchedModel::output_names() const noexcept
{
  return _impl->_active->output_names();
}

const std::vector<int> &
DispatchedModel::input_sizes() const noexcept
{
  return _impl->_active->input_sizes();
}

const std::vector<int> &
DispatchedModel::output_sizes() const noexcept
{
  return _impl->_active->output_sizes();
}

std::map<std::string, at::Tensor> &
DispatchedModel::named_parameters() noexcept
{
  return _impl->_active->named_parameters();
}

const std::map<std::string, at::Tensor> &
DispatchedModel::named_parameters() const noexcept
{
  return _impl->_active->named_parameters();
}

at::Device
DispatchedModel::device() const noexcept
{
  return _impl->_active->device();
}

at::ScalarType
DispatchedModel::dtype() const noexcept
{
  return _impl->_active->dtype();
}
} // namespace neml2::aoti
