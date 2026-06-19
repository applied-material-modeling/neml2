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

#include <condition_variable>
#include <cstddef>
#include <mutex>

#include <c10/core/Device.h>

#include "neml2/csrc/aoti/aoti_export.h"
#include "neml2/csrc/dispatchers/WorkScheduler.h"

namespace neml2::aoti
{
/**
 * @brief Base for asynchronous, multi-device schedulers.
 *
 * Provides the blocking load-tracking coordination shared by every async
 * policy: a mutex + condition variable, and the four-call protocol the
 * @ref DispatchedModel async pool drives against --
 * `schedule_work` / `dispatched_work` / `completed_work` / `wait_for_completion`.
 * Subclasses supply only the device-assignment *policy* through the `*_impl`
 * hooks (all invoked while `_mutex` is held). `StaticHybridScheduler` is the
 * concrete policy.
 *
 * @ref DispatchedModel picks the async pool over the synchronous loop by
 * `dynamic_cast`-ing its scheduler to `AsyncScheduler`.
 */
class AOTI_EXPORT AsyncScheduler : public WorkScheduler
{
public:
  /// Block until some device has spare capacity, then set @p dev / @p n to the
  /// next chunk's target device and the chunk size that device will accept. The
  /// caller clamps @p n to the work it has left.
  void schedule_work(at::Device & dev, std::size_t & n);

  /// Record that @p n items were dispatched to @p dev (raises its load).
  void dispatched_work(at::Device dev, std::size_t n);

  /// Record that @p n items finished on @p dev (lowers its load + wakes
  /// `schedule_work` / `wait_for_completion` waiters).
  void completed_work(at::Device dev, std::size_t n);

  /// Block until every dispatched item has completed.
  void wait_for_completion();

protected:
  /// Pick the next (dev, n) if a device has spare capacity now; return false to
  /// keep waiting. Called under `_mutex`.
  virtual bool schedule_work_impl(at::Device & dev, std::size_t & n) const = 0;
  /// Apply a dispatch to the load bookkeeping. Called under `_mutex`.
  virtual void dispatched_work_impl(at::Device dev, std::size_t n) = 0;
  /// Apply a completion to the load bookkeeping. Called under `_mutex`.
  virtual void completed_work_impl(at::Device dev, std::size_t n) = 0;
  /// True when no dispatched work is outstanding. Called under `_mutex`.
  virtual bool all_work_completed() const = 0;

  mutable std::mutex _mutex;
  std::condition_variable _cv;
};
} // namespace neml2::aoti
