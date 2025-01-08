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

#include "neml2/dispatcher/WorkScheduler.h"

namespace neml2
{
/**
 * @brief A scheduler for multiple devices with static priority management
 *
 * The devices could have different priority, batch sizes, and capacities. The priorities are
 * determined at construction time and remain unchanged throughout the lifetime of the scheduler.
 */
class StaticHybridScheduler : public WorkScheduler
{
public:
  /**
   * The constructor takes a device list, along with the batch sizes, capacities, and priorities for
   * each device.
   *
   * The device list should be unique and non-empty. torch::kCPU can appear at most once. When
   * multiple cuda devices are present, each of them must correspond to a specific device ID.
   *
   * One or more batch size should be provided. If the number of batch sizes is one, the same batch
   * size is associated with all devices. Otherwise, the number of batch sizes should match the
   * number of devices.
   *
   * Similarly, zero or more capacities should be provided. If the capacity list is empty, the
   * default capacity is infinity (i.e., std::numeric_limits<std::size_t>::max()) for all devices;
   * if the number of capacities is one, the same capacity is associated with all devices;
   * otherwise, the number of capacities should match the number of devices.
   *
   * An optional list of priorities can be provided. The number of priorities should match the
   * number of devices. If no priorities are provided, the priorities are determined by the order
   * specified by the \p device_list. If duplicate priorities are provided, the corresponding
   * devices have equal priority, and their relative dispatch order is undefined.
   *
   * \note For developers, below is a summary of the construct of torch::Device:
   * torch::Device represents a compute device on which a tensor is located. A device is uniquely
   * identified by a type, which specifies the type of machine it is (e.g. CPU or CUDA GPU), and a
   * device index or ordinal, which identifies the specific compute device when there is more than
   * one of a certain type. The device index is optional, and in its defaulted state represents
   * (abstractly) "the current device". Further, there are two constraints on the value of the
   * device index, if one is explicitly stored:
   * 1. A negative index represents the current device, a non-negative index
   *    represents a specific, concrete device,
   * 2. When the device type is CPU, the device index must be zero.
   */
  StaticHybridScheduler(const std::vector<torch::Device> & device_list,
                        const std::vector<std::size_t> & batch_sizes,
                        const std::vector<std::size_t> & capacities = {},
                        const std::vector<std::size_t> & priorities = {});

  bool next(torch::Device &, std::size_t &) const override;

  void dispatched(torch::Device, std::size_t) override;

  void completed(torch::Device, std::size_t) override;

  struct DeviceStatus
  {
    DeviceStatus(torch::Device device,
                 std::size_t batch_size,
                 std::size_t capacity,
                 std::size_t priority)
      : device(device),
        batch_size(batch_size),
        capacity(capacity),
        priority(priority),
        load(0)
    {
    }

    torch::Device device;
    std::size_t batch_size;
    std::size_t capacity;
    std::size_t priority;
    std::size_t load;
  };

  const std::vector<DeviceStatus> & status() const { return _devices; }

private:
  /// Whether a CPU device is specified
  bool _cpu = false;

  /// Whether any CUDA device is specified
  bool _cuda = false;

  /// The devices to dispatch to (in order of priority)
  std::vector<DeviceStatus> _devices;
};
} // namespace neml2
