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

#include <limits>

#include "neml2/dispatchers/WorkScheduler.h"

namespace neml2
{
/**
 * @brief A very simple scheduler
 *
 * This schedule is simple in the sense that
 * - It dispatches to a single device
 * - It dispatches a fixed batch size
 * - It does not perform parallel communication with other ranks (if any) to determine the
 *   availability of the device
 */
class SimpleScheduler : public WorkScheduler
{
public:
  /**
   * @brief Construct a new Simple Scheduler object
   *
   * @param device Device on which to dispatch
   * @param batch_size The (fixed) batch size to dispatch each time
   * @param capacity The capacity of the device, i.e., the maximum number of work batches that can
   * be simultaneously handled by the device at any given time. The default capacity is set to the
   * maximum value of size_t
   */
  SimpleScheduler(Device device,
                  std::size_t batch_size,
                  std::size_t capacity = std::numeric_limits<std::size_t>::max());

  bool schedule_work(Device &, std::size_t &) const override;

  void dispatched_work(Device, std::size_t) override;

  void completed_work(Device, std::size_t) override;

private:
  /// The device to dispatch to
  Device _device;

  /// The batch size to dispatch
  std::size_t _batch_size;

  /// The capacity of the device
  std::size_t _capacity;

  /// Current load on the device
  std::size_t _load = 0;
};
} // namespace neml2
