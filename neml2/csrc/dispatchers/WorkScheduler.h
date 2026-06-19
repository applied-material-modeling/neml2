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

#include <c10/core/Device.h>

#include "neml2/csrc/aoti/aoti_export.h"

namespace neml2::aoti
{
/**
 * @brief Decides where work runs and how it is chunked.
 *
 * A scheduler answers two questions for the @ref DispatchedModel that owns it:
 * which device a workload is placed on, and how large each batch chunk is. This
 * is the synchronous, single-device-per-instance contract -- one scheduler
 * instance maps to exactly one device.
 *
 * The richer asynchronous load-balancing surface from the v2 dispatcher
 * (`schedule_work` / `dispatched_work` / `completed_work` / `wait_for_completion`
 * plus a multi-device hybrid policy) is intentionally **not** reproduced here;
 * it belongs to the deferred asynchronous thread-pool phase and would only add
 * dead interface in the synchronous path.
 */
class AOTI_EXPORT WorkScheduler
{
public:
  WorkScheduler() = default;

  /// Out-of-line (defined in WorkScheduler.cpp) so the vtable + typeinfo are
  /// emitted once in the shared library rather than weakly in every consumer.
  virtual ~WorkScheduler();

  // Non-copyable, non-movable: schedulers are held by shared_ptr and carry
  // device state that should not be silently duplicated.
  WorkScheduler(const WorkScheduler &) = delete;
  WorkScheduler(WorkScheduler &&) = delete;
  WorkScheduler & operator=(const WorkScheduler &) = delete;
  WorkScheduler & operator=(WorkScheduler &&) = delete;

  /// The concrete device this scheduler dispatches to.
  virtual at::Device device() const = 0;

  /// Number of batch elements per chunk along the leading (dim-0) batch axis.
  /// A value of 0 means "no chunking" -- run the whole batch in a single call.
  virtual std::size_t batch_size() const = 0;
};
} // namespace neml2::aoti
