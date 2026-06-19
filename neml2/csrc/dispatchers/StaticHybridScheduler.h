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
#include <string>
#include <vector>

#include <c10/core/Device.h>

#include "neml2/csrc/aoti/aoti_export.h"
#include "neml2/csrc/dispatchers/AsyncScheduler.h"

namespace neml2::aoti
{
/**
 * @brief Spread one batch across several devices at once (CPU + GPU(s)).
 *
 * A single @ref DispatchedModel instance dispatches concurrently to every
 * device in the pool via the async thread-per-device path. Assignment is greedy:
 * each `schedule_work` hands the next chunk to the highest-priority device that
 * still has spare capacity (`load + batch_size <= capacity`), so faster / higher-
 * priority devices are kept filled. Configured entirely in C++ (no `[Schedulers]`
 * HIT surface).
 *
 * Tune each device's `batch_size` first (e.g. with @ref SimpleScheduler), then
 * size `capacity` (how many in-flight chunks a device may hold) to overlap the
 * next chunk's host->device copy with the current chunk's compute.
 *
 * The pool runs one worker thread per device, and each device's AOTI graph
 * already saturates torch's intra-op (OpenMP) pool. Distinct hardware therefore
 * does not contend, but two CPU entries would oversubscribe the same cores --
 * so the pool admits **at most one CPU**; the rest must be distinct GPUs.
 */
class AOTI_EXPORT StaticHybridScheduler : public AsyncScheduler
{
public:
  struct Config
  {
    /// Devices to dispatch to, e.g. {"cpu", "cuda:0", "cuda:1"}. Must be
    /// distinct.
    std::vector<std::string> devices;
    /// Per-device chunk size (must be > 0). Length 1 broadcasts to all devices;
    /// otherwise it must match `devices`.
    std::vector<std::size_t> batch_sizes;
    /// Per-device max in-flight items. Empty defaults each to its `batch_size`;
    /// length 1 broadcasts; otherwise must match `devices`. Each capacity must
    /// be >= the device's batch_size (so a chunk can always be placed).
    std::vector<std::size_t> capacities;
    /// Per-device greedy priority (higher = filled first). Empty defaults all to
    /// 1.0; length 1 broadcasts; otherwise must match `devices`.
    std::vector<double> priorities;
  };

  explicit StaticHybridScheduler(const Config & config);

  std::vector<at::Device> devices() const override { return _devices; }

protected:
  bool schedule_work_impl(at::Device & dev, std::size_t & n) const override;
  void dispatched_work_impl(at::Device dev, std::size_t n) override;
  void completed_work_impl(at::Device dev, std::size_t n) override;
  bool all_work_completed() const override;

private:
  struct DeviceStatus
  {
    at::Device device;
    std::size_t batch_size;
    std::size_t capacity;
    double priority;
    std::size_t load = 0;
  };

  std::vector<DeviceStatus> _status;
  std::vector<at::Device> _devices; // cached for devices()
};
} // namespace neml2::aoti
