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

#include "neml2/dispatcher/StaticHybridScheduler.h"
#include "neml2/misc/error.h"

namespace neml2
{
StaticHybridScheduler::StaticHybridScheduler(const std::vector<torch::Device> & device_list,
                                             const std::vector<std::size_t> & batch_sizes,
                                             const std::vector<std::size_t> & capacities,
                                             const std::vector<std::size_t> & priorities)
{
  // First pass:
  // - Check if any CPU device is present
  // - Check if any CUDA device is present
  // - Make sure no more than one CPU device is present
  for (const auto & device : device_list)
    if (device.is_cpu())
    {
      neml_assert(!_cpu, "Multiple CPU devices are not allowed");
      _cpu = true;
    }
    else if (device.is_cuda())
      _cuda = true;
    else
      neml_assert(false, "Unsupported device type: ", device);

  // Second pass:
  // - If multiple CUDA devices are present, make sure each CUDA device has a concrete
  //   (nonnegative), unique device ID
  bool has_multiple_cuda_devices = _cpu ? device_list.size() > 2 : device_list.size() > 1;
  if (has_multiple_cuda_devices)
  {
    std::set<torch::DeviceIndex> cuda_device_ids;
    for (const auto & device : device_list)
    {
      if (device.is_cpu())
        continue;
      auto device_id = device.index();
      neml_assert(device_id >= 0, "Device ID must be nonnegative");
      neml_assert(cuda_device_ids.find(device_id) == cuda_device_ids.end(),
                  "Device ID must be unique. Found duplicate: ",
                  device_id);
      cuda_device_ids.insert(device_id);
    }
  }

  // Expand batch size if necessary
  auto batch_sizes_expand = batch_sizes;
  if (batch_sizes.size() == 1)
    batch_sizes_expand.resize(device_list.size(), batch_sizes[0]);
  else
    neml_assert(batch_sizes.size() == device_list.size(),
                "Number of batch sizes must either be one or match the number of devices.");

  // Expand capacity if necessary
  auto capacities_expand = capacities;
  if (capacities.empty())
    capacities_expand.resize(device_list.size(), std::numeric_limits<std::size_t>::max());
  else if (capacities.size() == 1)
    capacities_expand.resize(device_list.size(), capacities[0]);
  else
    neml_assert(capacities.size() == device_list.size(),
                "Number of capacities must either be zero, one, or match the number of devices.");

  // Expand priorities if necessary
  auto priorities_expand = priorities;
  if (priorities.empty())
    priorities_expand.resize(device_list.size(), 0);
  else
    neml_assert(priorities.size() == device_list.size(),
                "Number of priorities must match the number of devices.");

  // Construct the device status list
  for (std::size_t i = 0; i < device_list.size(); ++i)
    _devices.emplace_back(
        device_list[i], batch_sizes_expand[i], capacities_expand[i], priorities_expand[i]);

  // Sort devices by priority (if provided)
  if (!priorities.empty())
    std::sort(_devices.begin(),
              _devices.end(),
              [](const DeviceStatus & a, const DeviceStatus & b)
              { return a.priority > b.priority; });
}

bool
StaticHybridScheduler::next(torch::Device & device, std::size_t & n) const
{
  for (const auto & i : _devices)
    if ((i.load + i.batch_size) <= i.capacity)
    {
      device = i.device;
      n = i.batch_size;
      return true;
    }
  return false;
}

void
StaticHybridScheduler::dispatched(torch::Device device, std::size_t n)
{
  for (auto & i : _devices)
    if (i.device == device)
    {
      i.load += n;
      // TODO: Add an option to allow for oversubscription, maybe?
      neml_assert(i.load <= i.capacity, "Device oversubscribed");
      return;
    }

  neml_assert(false, "Device not found: ", device);
}

void
StaticHybridScheduler::completed(torch::Device device, std::size_t n)
{
  for (auto & i : _devices)
    if (i.device == device)
    {
      neml_assert(i.load >= n, "Device load underflow");
      i.load -= n;
      return;
    }

  neml_assert(false, "Device not found: ", device);
}
} // namespace neml2
