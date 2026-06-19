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

#include <set>

#include "neml2/csrc/aoti/assertions.h"
#include "neml2/csrc/dispatchers/StaticHybridScheduler.h"

namespace neml2::aoti
{
namespace
{
// Expand a per-device config vector: length 1 broadcasts to n; length n passes
// through; anything else is an error.
template <typename T>
std::vector<T>
broadcast(const std::vector<T> & v, std::size_t n, const char * what)
{
  _assert(v.size() == 1 || v.size() == n,
          "StaticHybridScheduler: `",
          what,
          "` must have length 1 (broadcast) or match `devices` (",
          n,
          "); got ",
          v.size(),
          ".");
  if (v.size() == n)
    return v;
  return std::vector<T>(n, v.front());
}
} // namespace

StaticHybridScheduler::StaticHybridScheduler(const Config & config)
{
  const auto n = config.devices.size();
  _assert(n > 0, "StaticHybridScheduler: `devices` must be non-empty.");
  _assert(!config.batch_sizes.empty(), "StaticHybridScheduler: `batch_sizes` must be non-empty.");

  const auto batch_sizes = broadcast(config.batch_sizes, n, "batch_sizes");
  const auto capacities =
      config.capacities.empty() ? batch_sizes : broadcast(config.capacities, n, "capacities");
  const auto priorities = config.priorities.empty() ? std::vector<double>(n, 1.0)
                                                    : broadcast(config.priorities, n, "priorities");

  std::set<std::string> seen;
  std::size_t cpu_count = 0;
  _status.reserve(n);
  _devices.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    _assert(seen.insert(config.devices[i]).second,
            "StaticHybridScheduler: duplicate device '",
            config.devices[i],
            "'. Each device may appear at most once.");
    // One worker thread per device runs that device's AOTI graph, which itself
    // saturates torch's intra-op (OpenMP) pool with ~ncpu threads. Two workers
    // both targeting the CPU would oversubscribe the same cores for no gain, so
    // a hybrid pool admits at most one CPU; the rest must be distinct GPUs (use
    // explicit indices: cuda:0, cuda:1, ...).
    if (parse_device(config.devices[i]).is_cpu())
      _assert(++cpu_count == 1,
              "StaticHybridScheduler: more than one CPU device requested. A hybrid pool "
              "may include at most one CPU (concurrent CPU graphs only oversubscribe the "
              "intra-op thread pool); use one CPU plus distinct GPUs.");
    _assert(batch_sizes[i] > 0,
            "StaticHybridScheduler: device '",
            config.devices[i],
            "' has batch_size 0; hybrid chunks must be positive.");
    _assert(capacities[i] >= batch_sizes[i],
            "StaticHybridScheduler: device '",
            config.devices[i],
            "' capacity (",
            capacities[i],
            ") is smaller than its batch_size (",
            batch_sizes[i],
            "); a chunk could never be placed.");
    const auto dev = parse_device(config.devices[i]);
    _status.push_back({dev, batch_sizes[i], capacities[i], priorities[i], /*load=*/0});
    _devices.push_back(dev);
  }
}

bool
StaticHybridScheduler::schedule_work_impl(at::Device & dev, std::size_t & n) const
{
  // Greedy: among devices with spare capacity for another chunk, pick the
  // highest priority (first wins on ties, preserving config order).
  const DeviceStatus * best = nullptr;
  for (const auto & s : _status)
    if (s.load + s.batch_size <= s.capacity)
      if (best == nullptr || s.priority > best->priority)
        best = &s;

  if (best == nullptr)
    return false;
  dev = best->device;
  n = best->batch_size;
  return true;
}

void
StaticHybridScheduler::dispatched_work_impl(at::Device dev, std::size_t n)
{
  for (auto & s : _status)
    if (s.device == dev)
    {
      s.load += n;
      return;
    }
  _assert(false, "StaticHybridScheduler: dispatched_work for unknown device '", dev.str(), "'.");
}

void
StaticHybridScheduler::completed_work_impl(at::Device dev, std::size_t n)
{
  for (auto & s : _status)
    if (s.device == dev)
    {
      _assert(s.load >= n,
              "StaticHybridScheduler: completed_work (",
              n,
              ") exceeds the outstanding load (",
              s.load,
              ") on device '",
              dev.str(),
              "'.");
      s.load -= n;
      return;
    }
  _assert(false, "StaticHybridScheduler: completed_work for unknown device '", dev.str(), "'.");
}

bool
StaticHybridScheduler::all_work_completed() const
{
  for (const auto & s : _status)
    if (s.load != 0)
      return false;
  return true;
}
} // namespace neml2::aoti
