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

#include <mutex>
#include <condition_variable>

#include "neml2/base/NEML2Object.h"
#include "neml2/misc/types.h"
#include "neml2/misc/TracingInterface.h"

namespace neml2
{
/**
 * @brief Scheduler for work dispatching
 *
 * The scheduler is responsible for determining
 * 1. The amount (number of batches) of work to be dispatched next
 * 2. Where (e.g., to which device) the next batch of work should be dispatched
 *
 * The scheduler is also responsible for updating its internal state when work is dispatched.
 *
 * @see WorkGenerator, WorkDispatcher
 */
class WorkScheduler : public NEML2Object, public TracingInterface
{
public:
  /// Options for the scheduler
  static OptionSet expected_options();

  /**
   * @brief Construct a new WorkScheduler object
   *
   * @param options Options for the scheduler
   */
  WorkScheduler(const OptionSet & options);

  /**
   * @brief Determine the device and batch size for the next dispatch
   *
   * This is blocking until work can be scheduled.
   */
  void schedule_work(Device &, std::size_t &);

  /// Update the scheduler with the dispatch of the last batch
  void dispatched_work(Device, std::size_t);

  /// Update the scheduler with the completion of the last batch
  void completed_work(Device, std::size_t);

  /// Wait for all work to complete
  void wait_for_completion();

  /// Device options
  virtual std::vector<Device> devices() const { return std::vector<Device>(); }

protected:
  /// Implementation of the work scheduling
  virtual bool schedule_work_impl(Device &, std::size_t &) const = 0;

  /// Update the scheduler with the dispatch of the last batch
  virtual void dispatched_work_impl(Device, std::size_t) = 0;

  /// Update the scheduler with the completion of the last batch
  virtual void completed_work_impl(Device, std::size_t) = 0;

  /// Check if all work has been completed
  virtual bool all_work_completed() const = 0;

  /// Condition variable for the scheduling thread
  std::condition_variable _condition;

private:
  /// Mutex for this scheduler
  std::mutex _mutex;
};
} // namespace neml2
