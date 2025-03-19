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

#include "neml2/dispatchers/WorkScheduler.h"
#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"

#include "timpi/communicator.h"

namespace neml2
{
/**
 * @brief A simple MPI scheduler
 *
 * Each rank dispatches to a single device, selected based on its processor ID.
 *
 * It dispataches a fixed batch size, up to (optionally) the provided capacity.  The default
 * capacity is the batch size.
 *
 * The number of devices in the list must equal the local communicator size.  The list must not
 * include the cpu.
 */
class SimpleMPIScheduler : public WorkScheduler
{
public:
  /// Options for the scheduler
  static OptionSet expected_options();

  /**
   * @brief Construct a new WorkScheduler object
   *
   * @param options Options for the scheduler
   */
  SimpleMPIScheduler(const OptionSet & options);

  /// @brief  Check the device list and coordinate this rank's device
  void setup() override;

  virtual std::vector<Device> devices() const override
  {
    return {_available_devices[_device_index]};
  }

  virtual TIMPI::Communicator & comm() { return _comm; }

  virtual void set_comm(TIMPI::Communicator comm) { _comm.duplicate(comm); }

protected:
  bool schedule_work_impl(Device &, std::size_t &) const override;
  void dispatched_work_impl(Device, std::size_t) override;
  void completed_work_impl(Device, std::size_t) override;
  bool all_work_completed() const override;

private:
  /// Figure out which device to use
  void determine_my_device();

  /// The devices to dispatch to
  std::vector<Device> _available_devices;

  /// The batch size to dispatch
  std::size_t _batch_size;

  /// The capacity of the device
  std::size_t _capacity;

  /// This rank's device
  size_t _device_index;

  /// Global communicator to use to split
  TIMPI::Communicator _comm;

  /// Current load on the device
  std::size_t _load = 0;
};

} // namespace neml2
