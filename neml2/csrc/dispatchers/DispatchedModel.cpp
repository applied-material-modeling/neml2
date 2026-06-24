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

#include <condition_variable>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/dispatchers/AsyncScheduler.h"
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
    classify_scheduler();

    // One Model per scheduler device, from the matching <device-type>/ subfolder,
    // pinned to that device's concrete index (e.g. one cuda artifact -> cuda:0,
    // cuda:1). The first device is the primary (metadata + master parameters).
    const auto devs = _scheduler->devices();
    _assert(!devs.empty(), "DispatchedModel: scheduler reported no devices.");
    for (const auto & dev : devs)
    {
      const auto subdir = artifact_root / (dev.is_cuda() ? "cuda" : "cpu");
      auto model = std::make_unique<Model>(find_meta(subdir), dev);
      if (_active == nullptr)
        _active = model.get();
      _models.emplace(dev.str(), std::move(model));
    }
    start_pool_if_async();
  }

  Impl(std::unique_ptr<Model> model, std::shared_ptr<WorkScheduler> scheduler)
    : _scheduler(std::move(scheduler))
  {
    _assert(static_cast<bool>(_scheduler), "DispatchedModel: scheduler must not be null.");
    _assert(static_cast<bool>(model), "DispatchedModel: model must not be null.");
    classify_scheduler();
    _assert(_sync != nullptr,
            "DispatchedModel(Model, scheduler): only a synchronous, single-device scheduler is "
            "supported here; load multi-device hybrids from an artifact root.");
    _assert(model->device().type() == _sync->device().type(),
            "DispatchedModel: scheduler device '",
            _sync->device().str(),
            "' does not match the provided model's device '",
            model->device().str(),
            "'.");
    _active = model.get();
    _models.emplace(model->device().str(), std::move(model));
  }

  ~Impl() { stop_pool(); }

  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  // --- dispatch entry points (sync vs async chosen by scheduler type) --------

  std::map<std::string, at::Tensor> forward(const std::map<std::string, at::Tensor> & inputs)
  {
    _assert(!inputs.empty(), "DispatchedModel::forward: inputs are empty.");
    sync_params();
    const auto in_device = inputs.begin()->second.device();
    const int64_t b = infer_batch_size(inputs);

    // Per chunk: slice inputs + any batched parameters to the chunk's rows, run
    // on the chunk's device, move the result back to the input device.
    auto chunk_fn = [&](at::Device d, int64_t s, int64_t cnt)
    {
      auto in = to_device(slice_batch(inputs, s, cnt), d);
      auto ov = chunk_param_overrides(s, cnt, b, d);
      return to_device(_models.at(d.str())->forward(in, ov), in_device);
    };

    if (_async != nullptr)
      return cat_batch(run_async<std::map<std::string, at::Tensor>>(b, chunk_fn));

    const int64_t chunk = chunk_extent(b);
    if (chunk >= b && _active->device() == in_device)
      return _active->forward(inputs); // fast path (full batched param used as-is)

    std::vector<std::map<std::string, at::Tensor>> results;
    for (int64_t s = 0; s < b; s += chunk)
      results.push_back(chunk_fn(_active->device(), s, std::min(chunk, b - s)));
    return cat_batch(results);
  }

  std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
  jvp(const std::map<std::string, at::Tensor> & inputs,
      const std::map<std::string, at::Tensor> & tangents)
  {
    _assert(!inputs.empty(), "DispatchedModel::jvp: inputs are empty.");
    sync_params();
    const auto in_device = inputs.begin()->second.device();
    const int64_t b = infer_batch_size(inputs);

    using Pair = std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>;
    auto chunk_fn = [&](at::Device d, int64_t s, int64_t cnt) -> Pair
    {
      auto in = to_device(slice_batch(inputs, s, cnt), d);
      auto tan = to_device(slice_batch(tangents, s, cnt), d);
      auto ov = chunk_param_overrides(s, cnt, b, d);
      auto [out, jout] = _models.at(d.str())->jvp(in, tan, ov);
      return {to_device(out, in_device), to_device(jout, in_device)};
    };

    std::vector<Pair> chunks;
    if (_async != nullptr)
      chunks = run_async<Pair>(b, chunk_fn);
    else
    {
      const int64_t chunk = chunk_extent(b);
      if (chunk >= b && _active->device() == in_device)
        return _active->jvp(inputs, tangents); // fast path
      for (int64_t s = 0; s < b; s += chunk)
        chunks.push_back(chunk_fn(_active->device(), s, std::min(chunk, b - s)));
    }

    std::vector<std::map<std::string, at::Tensor>> outs, jouts;
    outs.reserve(chunks.size());
    jouts.reserve(chunks.size());
    for (auto & c : chunks)
    {
      outs.push_back(std::move(c.first));
      jouts.push_back(std::move(c.second));
    }
    return {cat_batch(outs), cat_batch(jouts)};
  }

  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  jacobian(const std::map<std::string, at::Tensor> & inputs)
  {
    _assert(!inputs.empty(), "DispatchedModel::jacobian: inputs are empty.");
    sync_params();
    const auto in_device = inputs.begin()->second.device();
    const int64_t b = infer_batch_size(inputs);

    using Pair = std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>;
    auto chunk_fn = [&](at::Device d, int64_t s, int64_t cnt) -> Pair
    {
      auto in = to_device(slice_batch(inputs, s, cnt), d);
      auto ov = chunk_param_overrides(s, cnt, b, d);
      auto [out, j] = _models.at(d.str())->jacobian(in, ov);
      return {to_device(out, in_device), to_device_nested(j, in_device)};
    };

    std::vector<Pair> chunks;
    if (_async != nullptr)
      chunks = run_async<Pair>(b, chunk_fn);
    else
    {
      const int64_t chunk = chunk_extent(b);
      if (chunk >= b && _active->device() == in_device)
        return _active->jacobian(inputs); // fast path
      for (int64_t s = 0; s < b; s += chunk)
        chunks.push_back(chunk_fn(_active->device(), s, std::min(chunk, b - s)));
    }

    std::vector<std::map<std::string, at::Tensor>> outs;
    std::vector<VariablePairJacobian> jparts;
    outs.reserve(chunks.size());
    jparts.reserve(chunks.size());
    for (auto & c : chunks)
    {
      outs.push_back(std::move(c.first));
      jparts.push_back(std::move(c.second));
    }
    // Per-pair reassembly is base-shape-aware so a batch-independent block
    // (returned unbatched by the single-forward fast path) is passed through
    // rather than concatenated across chunks. Build name -> base-ndim maps from
    // the active model's declared layout.
    std::map<std::string, int64_t> out_base_ndim, in_base_ndim;
    {
      const auto & onames = _active->output_names();
      const auto & oshapes = _active->output_base_shapes();
      for (std::size_t k = 0; k < onames.size(); ++k)
        out_base_ndim[onames[k]] = static_cast<int64_t>(oshapes[k].size());
      const auto & inames = _active->input_names();
      const auto & ishapes = _active->input_base_shapes();
      for (std::size_t k = 0; k < inames.size(); ++k)
        in_base_ndim[inames[k]] = static_cast<int64_t>(ishapes[k].size());
    }
    return {cat_batch(outs), cat_batch_nested(jparts, out_base_ndim, in_base_ndim)};
  }

  std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
  param_jacobian(const std::map<std::string, at::Tensor> & inputs)
  {
    _assert(!inputs.empty(), "DispatchedModel::param_jacobian: inputs are empty.");
    sync_params();
    const auto in_device = inputs.begin()->second.device();
    const int64_t b = infer_batch_size(inputs);

    using Pair = std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>;
    auto chunk_fn = [&](at::Device d, int64_t s, int64_t cnt) -> Pair
    {
      auto in = to_device(slice_batch(inputs, s, cnt), d);
      auto ov = chunk_param_overrides(s, cnt, b, d);
      auto [out, p] = _models.at(d.str())->param_jacobian(in, ov);
      return {to_device(out, in_device), to_device_nested(p, in_device)};
    };

    std::vector<Pair> chunks;
    if (_async != nullptr)
      chunks = run_async<Pair>(b, chunk_fn);
    else
    {
      const int64_t chunk = chunk_extent(b);
      if (chunk >= b && _active->device() == in_device)
        return _active->param_jacobian(inputs); // fast path
      for (int64_t s = 0; s < b; s += chunk)
        chunks.push_back(chunk_fn(_active->device(), s, std::min(chunk, b - s)));
    }

    std::vector<std::map<std::string, at::Tensor>> outs;
    std::vector<VariablePairJacobian> pparts;
    outs.reserve(chunks.size());
    pparts.reserve(chunks.size());
    for (auto & c : chunks)
    {
      outs.push_back(std::move(c.first));
      pparts.push_back(std::move(c.second));
    }
    // Same base-shape-aware reassembly as jacobian(): the "column" axis here is
    // the parameter, so the inner base-ndim map is keyed by parameter qname. A
    // batch-independent block (returned unbatched) is passed through, not
    // concatenated. Parameter base shapes come from the active model's promoted
    // parameter values (each stored at its (*param_base) shape).
    std::map<std::string, int64_t> out_base_ndim, param_base_ndim;
    {
      const auto & onames = _active->output_names();
      const auto & oshapes = _active->output_base_shapes();
      for (std::size_t k = 0; k < onames.size(); ++k)
        out_base_ndim[onames[k]] = static_cast<int64_t>(oshapes[k].size());
      // NATURAL base ndim, not the stored tensor's dim(): a batched parameter
      // `(B, *base)` would otherwise report an inflated trailing-ndim and
      // mis-classify its `(*B, *out_base, *param_base)` block as
      // batch-independent in cat_batch_nested (dropping all but the first chunk).
      for (const auto & [q, base] : _active->parameter_base_shapes())
        param_base_ndim[q] = static_cast<int64_t>(base.size());
    }
    return {cat_batch(outs), cat_batch_nested(pparts, out_base_ndim, param_base_ndim)};
  }

  std::map<std::string, at::Tensor> param_vjp(const std::map<std::string, at::Tensor> & inputs,
                                              const std::map<std::string, at::Tensor> & cotangents)
  {
    _assert(!inputs.empty(), "DispatchedModel::param_vjp: inputs are empty.");
    sync_params();
    const auto in_device = inputs.begin()->second.device();
    const int64_t b = infer_batch_size(inputs);

    using Ret = std::map<std::string, at::Tensor>;
    auto chunk_fn = [&](at::Device d, int64_t s, int64_t cnt) -> Ret
    {
      // Slice inputs, output cotangents, AND any batched parameter to the chunk's
      // rows; the per-device param_vjp collapses each gradient to the (per-chunk)
      // stored/override shape -- per-element for a batched parameter, summed for a
      // global one.
      auto in = to_device(slice_batch(inputs, s, cnt), d);
      auto cot = to_device(slice_batch(cotangents, s, cnt), d);
      auto ov = chunk_param_overrides(s, cnt, b, d);
      return to_device(_models.at(d.str())->param_vjp(in, cot, ov), in_device);
    };

    std::vector<Ret> chunks;
    if (_async != nullptr)
      chunks = run_async<Ret>(b, chunk_fn);
    else
    {
      const int64_t chunk = chunk_extent(b);
      if (chunk >= b && _active->device() == in_device)
        return _active->param_vjp(inputs, cotangents); // fast path
      for (int64_t s = 0; s < b; s += chunk)
        chunks.push_back(chunk_fn(_active->device(), s, std::min(chunk, b - s)));
    }
    // Adjoint stitch, per parameter: a BATCHED (per-batch-element) parameter's
    // per-chunk gradients are CONCATENATED back to the full batch; a GLOBAL
    // parameter's are SUMMED (each chunk already reduced its own slice). Mirrors
    // the per-device collapse so the dispatched result equals a single shot.
    _assert(!chunks.empty(), "DispatchedModel::param_vjp: no chunks produced.");
    if (chunks.size() == 1)
      return chunks.front();
    std::map<std::string, at::Tensor> out;
    const auto & bases = _active->parameter_base_shapes();
    const auto & master = _active->named_parameters();
    for (const auto & [q, first] : chunks.front())
    {
      auto bit = bases.find(q);
      const int64_t base_ndim = (bit != bases.end()) ? static_cast<int64_t>(bit->second.size()) : 0;
      auto mit = master.find(q);
      const bool batched =
          mit != master.end() && (mit->second.dim() - base_ndim >= 1) && (mit->second.size(0) == b);
      std::vector<at::Tensor> parts;
      parts.reserve(chunks.size());
      for (const auto & c : chunks)
      {
        auto it = c.find(q);
        _assert(it != c.end(), "DispatchedModel::param_vjp: key '", q, "' missing from a chunk.");
        parts.push_back(it->second);
      }
      if (batched)
        out.emplace(q, at::cat(parts, /*dim=*/0));
      else
      {
        at::Tensor acc = parts.front();
        for (std::size_t k = 1; k < parts.size(); ++k)
          acc = acc + parts[k];
        out.emplace(q, std::move(acc));
      }
    }
    return out;
  }

  void set_solver_config(const SolverConfig & config)
  {
    for (auto & [name, m] : _models)
      m->set_solver_config(config);
  }

  // Master promoted-parameter map (the primary device copy). Marking it dirty
  // on mutable access broadcasts it to the other device copies before the next
  // dispatch.
  std::map<std::string, at::Tensor> & params_mut()
  {
    _params_dirty = true;
    return _active->named_parameters();
  }
  const std::map<std::string, at::Tensor> & params() const { return _active->named_parameters(); }

  void set_parameter(const std::string & name, const at::Tensor & value)
  {
    // Update the master (active-device) slot; the per-device copies are
    // re-synced from it on the next dispatched call.
    _active->set_parameter(name, value);
    _params_dirty = true;
  }

  Model * active() const { return _active; }

private:
  void classify_scheduler()
  {
    _async = dynamic_cast<AsyncScheduler *>(_scheduler.get());
    _sync = dynamic_cast<SyncScheduler *>(_scheduler.get());
    _assert(_async != nullptr || _sync != nullptr,
            "DispatchedModel: scheduler is neither a SyncScheduler nor an AsyncScheduler.");
  }

  /// Sync chunk extent along dim 0: the scheduler's batch size, clamped to the
  /// whole batch (0 => no chunking).
  int64_t chunk_extent(int64_t b) const
  {
    const auto n = _sync->batch_size();
    if (n == 0 || static_cast<int64_t>(n) >= b)
      return b;
    return static_cast<int64_t>(n);
  }

  /// Broadcast the master promoted params to every other device copy. No-op for
  /// a single device, an empty (fully-baked) param set, or an unmutated master.
  ///
  /// Uses slot ASSIGNMENT (not in-place `copy_`): a batched parameter set via
  /// `set_parameter` re-shapes the master slot from `(*base)` to `(B, *base)`,
  /// which an in-place `copy_` into the device copy's original-shaped slot would
  /// reject. The per-device copy of a batched parameter is then never actually
  /// read -- each dispatched chunk passes its own sliced override (see
  /// `chunk_param_overrides`) -- but keeping the copies shape-consistent with the
  /// master keeps the unbatched path and any direct device-model use correct.
  void sync_params()
  {
    if (!_params_dirty || _models.size() <= 1)
    {
      _params_dirty = false;
      return;
    }
    const auto & master = _active->named_parameters();
    for (auto & [name, m] : _models)
    {
      if (m.get() == _active)
        continue;
      const auto dev = m->device();
      auto & dst = m->named_parameters();
      for (const auto & [k, v] : master)
      {
        auto it = dst.find(k);
        _assert(it != dst.end(),
                "DispatchedModel: promoted parameter '",
                k,
                "' is missing on device '",
                name,
                "'.");
        it->second = (v.device() == dev) ? v : v.to(dev); // cross-device slot assign
      }
    }
    _params_dirty = false;
  }

  /// Per-chunk promoted-parameter overrides for the chunk `[s, s+cnt)`: each
  /// BATCHED stored parameter (a leading dim beyond its natural base, sized to
  /// the call batch `B`) narrowed to the chunk's rows and moved to device `d`.
  /// Unbatched / scalar parameters are omitted -- the per-device Model uses its
  /// own synced copy (broadcast in-graph to the chunk batch). Empty when nothing
  /// is batched, so the unbatched path costs nothing. Read-only on the master, so
  /// safe to call concurrently from the async workers.
  std::map<std::string, at::Tensor>
  chunk_param_overrides(int64_t s, int64_t cnt, int64_t b, at::Device d) const
  {
    std::map<std::string, at::Tensor> ov;
    const auto & params = _active->named_parameters();
    const auto & bases = _active->parameter_base_shapes();
    for (const auto & [q, t] : params)
    {
      auto bit = bases.find(q);
      const int64_t base_ndim = (bit != bases.end()) ? static_cast<int64_t>(bit->second.size()) : 0;
      // Batched iff it carries a leading dim past its base AND that dim is the
      // call batch. A size-1 / unbatched leading dim broadcasts -- leave it to
      // the stored copy.
      if (t.dim() - base_ndim < 1 || t.size(0) != b)
        continue;
      auto sl = t.narrow(0, s, cnt);
      ov.emplace(q, sl.device() == d ? sl : sl.to(d));
    }
    return ov;
  }

  // --- async thread-per-device pool ------------------------------------------

  /// Dispatch the b-row batch across the async pool: pull (device, chunk) from
  /// the scheduler, enqueue to the device's worker, and collect results in
  /// dispatch order. Blocks until every chunk completes.
  ///
  /// Exception safety. A chunk that throws (Newton non-convergence, a shape /
  /// device mismatch, an out-of-memory transfer, ...) must not escape its worker
  /// thread -- a C++ exception leaving a `std::thread` callable calls
  /// `std::terminate`. Each worker instead captures its failure and *still*
  /// drains the scheduler, so `wait_for_completion` can never deadlock on a
  /// stranded load. We then always wait for every in-flight chunk before
  /// deciding what to throw, so the decision sees the *complete* set of
  /// failures (concurrent chunks on different devices can fail at once):
  ///   - no failures -> return the results;
  ///   - exactly one -> re-throw it verbatim, preserving its dynamic type so a
  ///     consumer's `catch (const ConvergenceError &)` still matches;
  ///   - several     -> throw an `AggregateError` carrying them all. It reports
  ///     `recoverable()` only if every sub-error is recoverable, so a lone fatal
  ///     among otherwise-recoverable failures still forces a hard stop.
  /// Either way the pool + scheduler are left clean for the next call.
  template <typename R, typename Fn>
  std::vector<R> run_async(int64_t b, Fn chunk_fn)
  {
    std::vector<R> results;
    std::vector<std::exception_ptr> errors; // parallel to `results`; null == ok
    std::mutex results_mutex;               // guards `results`, `errors`, `failed`
    bool failed = false;
    // A failure on the dispatch thread (scheduling policy / enqueue), kept apart
    // so it sorts ahead of the per-chunk errors in the aggregate.
    std::exception_ptr dispatch_error;

    try
    {
      for (int64_t start = 0; start < b;)
      {
        // Once any chunk has failed, stop scheduling new work; the in-flight
        // chunks still drain below so we decide on the complete error set.
        {
          std::lock_guard<std::mutex> lock(results_mutex);
          if (failed)
            break;
        }

        at::Device dev = at::kCPU;
        std::size_t n = 0;
        _async->schedule_work(dev, n); // blocks until a device has spare capacity
        // We may have blocked above precisely until the chunk whose failure we
        // are now reacting to completed (it frees the capacity we waited on), so
        // re-check before committing this chunk -- otherwise a single-device pool
        // would dispatch one extra doomed chunk per failure.
        {
          std::lock_guard<std::mutex> lock(results_mutex);
          if (failed)
            break;
        }
        const int64_t count = std::min<int64_t>(static_cast<int64_t>(n), b - start);

        std::size_t idx = 0;
        {
          std::lock_guard<std::mutex> lock(results_mutex);
          results.emplace_back();
          errors.emplace_back();
          idx = results.size() - 1;
        }

        const at::Device d = dev;
        const int64_t s = start;
        // `noexcept`: the body must never let an exception reach the worker's
        // top-level call site -- it captures any failure and always drains the
        // scheduler load it was dispatched against.
        auto task = [this,
                     &results,
                     &errors,
                     &results_mutex,
                     &failed,
                     &chunk_fn,
                     d,
                     s,
                     count,
                     idx]() noexcept
        {
          try
          {
            R r = chunk_fn(d, s, count);
            std::lock_guard<std::mutex> lock(results_mutex);
            results[idx] = std::move(r);
          }
          catch (...)
          {
            std::lock_guard<std::mutex> lock(results_mutex);
            errors[idx] = std::current_exception();
            failed = true;
          }
          // Always: balances `dispatched_work` below so a failed chunk cannot
          // strand `wait_for_completion`.
          _async->completed_work(d, static_cast<std::size_t>(count));
        };

        _async->dispatched_work(d, static_cast<std::size_t>(count));
        try
        {
          enqueue(d, std::move(task));
        }
        catch (...)
        {
          // Failed to make the task runnable (e.g. OOM growing the queue): undo
          // the load we just added so the drain stays balanced, then let the
          // outer handler record it.
          _async->completed_work(d, static_cast<std::size_t>(count));
          throw;
        }
        start += count;
      }
    }
    catch (...)
    {
      // A failure on the dispatch thread itself (scheduling policy or enqueue).
      // Fall through to the drain regardless: the worker tasks capture
      // `results` / `errors` by reference, so none may outlive this frame.
      std::lock_guard<std::mutex> lock(results_mutex);
      if (!dispatch_error)
        dispatch_error = std::current_exception();
      failed = true;
    }

    // Wait for every in-flight chunk: no worker may outlive this frame, and we
    // need the complete failure set before deciding what to throw. After this
    // returns all workers are idle, so the vectors are safe to read unlocked.
    _async->wait_for_completion();

    // Gather failures in dispatch order (deterministic, independent of which
    // worker happened to finish first).
    std::vector<std::exception_ptr> failures;
    if (dispatch_error)
      failures.push_back(dispatch_error);
    for (auto & e : errors)
      if (e)
        failures.push_back(std::move(e));

    if (failures.empty())
      return results;
    if (failures.size() == 1)
      std::rethrow_exception(failures.front()); // exact dynamic type preserved
    throw AggregateError(std::move(failures));  // recoverable iff all are
  }

  void start_pool_if_async()
  {
    if (_async == nullptr)
      return;
    for (const auto & [name, m] : _models)
      _tasks[name]; // create each device's queue up front
    // Initialise torch's linalg backend before any threaded call
    // (https://github.com/pytorch/pytorch/issues/90613).
    at::linalg_inv(at::ones({1, 1}));
    for (const auto & [name, m] : _models)
      _threads.emplace_back([this, key = name] { worker_main(key); });
  }

  void worker_main(const std::string & key)
  {
    while (true)
    {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(_qmutex);
        _qcv.wait(lock, [&] { return _stop || !_tasks.at(key).empty(); });
        if (_stop && _tasks.at(key).empty())
          return;
        task = std::move(_tasks.at(key).front());
        _tasks.at(key).pop();
      }
      task();
    }
  }

  void enqueue(at::Device d, std::function<void()> task)
  {
    {
      std::lock_guard<std::mutex> lock(_qmutex);
      _tasks.at(d.str()).push(std::move(task));
    }
    _qcv.notify_all(); // the device-bound worker picks its own queue
  }

  void stop_pool()
  {
    if (_threads.empty())
      return;
    {
      std::lock_guard<std::mutex> lock(_qmutex);
      _stop = true;
    }
    _qcv.notify_all();
    for (auto & t : _threads)
      t.join();
    _threads.clear();
  }

  std::shared_ptr<WorkScheduler> _scheduler;
  SyncScheduler * _sync = nullptr;   // non-null for the sync path
  AsyncScheduler * _async = nullptr; // non-null for the async path

  // device-string -> Model (one per scheduler device). `_active` is the primary
  // (first device): metadata source + master promoted-parameter copy.
  std::map<std::string, std::unique_ptr<Model>> _models;
  Model * _active = nullptr;
  bool _params_dirty = false;

  // Async pool (unused / empty in the sync path).
  std::map<std::string, std::queue<std::function<void()>>> _tasks;
  std::mutex _qmutex;
  std::condition_variable _qcv;
  bool _stop = false;
  std::vector<std::thread> _threads;
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

// As with `Model`, the three ops run through `_guarded` so every exception that
// leaves the dispatcher is a neml2 Exception with a meaningful `recoverable()`.
// The async path already normalizes per-chunk failures (and aggregates
// concurrent ones via AggregateError); this also covers main-thread work outside
// the workers -- parameter sync, batch slicing, and the final concatenation.
std::map<std::string, at::Tensor>
DispatchedModel::forward(const std::map<std::string, at::Tensor> & inputs) const
{
  return _guarded([&] { return _impl->forward(inputs); });
}

std::pair<std::map<std::string, at::Tensor>, std::map<std::string, at::Tensor>>
DispatchedModel::jvp(const std::map<std::string, at::Tensor> & inputs,
                     const std::map<std::string, at::Tensor> & tangents) const
{
  return _guarded([&] { return _impl->jvp(inputs, tangents); });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
DispatchedModel::jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  return _guarded([&] { return _impl->jacobian(inputs); });
}

std::pair<std::map<std::string, at::Tensor>, VariablePairJacobian>
DispatchedModel::param_jacobian(const std::map<std::string, at::Tensor> & inputs) const
{
  return _guarded([&] { return _impl->param_jacobian(inputs); });
}

std::map<std::string, at::Tensor>
DispatchedModel::param_vjp(const std::map<std::string, at::Tensor> & inputs,
                           const std::map<std::string, at::Tensor> & cotangents) const
{
  return _guarded([&] { return _impl->param_vjp(inputs, cotangents); });
}

void
DispatchedModel::set_solver_config(const SolverConfig & config)
{
  _impl->set_solver_config(config);
}

const std::vector<std::string> &
DispatchedModel::input_names() const noexcept
{
  return _impl->active()->input_names();
}

const std::vector<std::string> &
DispatchedModel::output_names() const noexcept
{
  return _impl->active()->output_names();
}

const std::vector<std::vector<int64_t>> &
DispatchedModel::input_base_shapes() const noexcept
{
  return _impl->active()->input_base_shapes();
}

const std::vector<std::vector<int64_t>> &
DispatchedModel::output_base_shapes() const noexcept
{
  return _impl->active()->output_base_shapes();
}

std::map<std::string, at::Tensor> &
DispatchedModel::named_parameters() noexcept
{
  return _impl->params_mut();
}

const std::map<std::string, at::Tensor> &
DispatchedModel::named_parameters() const noexcept
{
  return _impl->params();
}

void
DispatchedModel::set_parameter(const std::string & name, const at::Tensor & value)
{
  _impl->set_parameter(name, value);
}

at::Device
DispatchedModel::device() const noexcept
{
  return _impl->active()->device();
}

at::ScalarType
DispatchedModel::dtype() const noexcept
{
  return _impl->active()->dtype();
}
} // namespace neml2::aoti
