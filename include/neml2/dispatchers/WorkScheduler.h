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

#include "neml2/base/NEML2Object.h"
#include "neml2/misc/types.h"

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
class WorkScheduler : public NEML2Object
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
   * @return true If work has been scheduled, i.e., there is a worker available
   * @return false If work cannot be scheduled, i.e., there is no worker available
   */
  virtual bool schedule_work(Device &, std::size_t &) const = 0;

  /// Update the schedule with the dispatch of the last batch
  virtual void dispatched_work(Device, std::size_t) = 0;

  /// Update the schedule with the completion of the last batch
  virtual void completed_work(Device, std::size_t) = 0;
};
} // namespace neml2
