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

#include "neml2/dispatchers/SimpleMPIScheduler.h"
#include "neml2/misc/assertions.h"

#include "hwloc.h"

#include <string>
#include <functional>

namespace neml2
{

register_NEML2_object(SimpleMPIScheduler);

OptionSet
SimpleMPIScheduler::expected_options()
{
  OptionSet options = WorkScheduler::expected_options();
  options.doc() =
      "Dispatch work to a single device selected based on processor ID in given batch sizes.";

  options.set<std::vector<Device>>("devices");
  options.set("devices").doc() = "List of devices to dispatch work to";

  options.set<std::size_t>("batch_size");
  options.set("batch_size").doc() = "Batch size";

  options.set<std::size_t>("capacity");
  options.set("capacity").doc() =
      "Maximum number of work items that can be dispatched, default to batch_size";

  return options;
}

SimpleMPIScheduler::SimpleMPIScheduler(const OptionSet & options)
  : WorkScheduler(options),
    _available_devices(options.get<std::vector<Device>>("devices")),
    _batch_size(options.get<std::size_t>("batch_size")),
    _capacity(options.get("capacity").user_specified() ? options.get<std::size_t>("capacity")
                                                       : _batch_size),
    _comm(TIMPI::Communicator(MPI_COMM_WORLD))
{
}

void
SimpleMPIScheduler::setup()
{
  WorkScheduler::setup();

  const auto & device_list = input_options().get<std::vector<Device>>("devices");

  // First pass:
  // - Prohibit any CPU
  // - Check if any CUDA device is present
  bool cuda = false;
  for (const auto & device : device_list)
  {
    neml_assert(!device.is_cpu(), "CPU device is not allowed in SimpleMPIScheduler");
    if (device.is_cuda())
      cuda = true;
    else
      neml_assert(false, "Unsupported device type: ", device);
  }

  // Second pass:
  // - If multiple CUDA devices are present, make sure each CUDA device has a concrete
  //   (nonnegative), unique device ID
  bool has_multiple_cuda_devices = device_list.size() > 1 && cuda;
  if (has_multiple_cuda_devices)
  {
    std::set<DeviceIndex> cuda_device_ids;
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

  determine_my_device();
}

void
SimpleMPIScheduler::determine_my_device()
{
  // Get the hostname via hwloc and hash it into an MPI_INT
  hwloc_topology_t topology;
  hwloc_obj_t machine;

  hwloc_topology_init(&topology); // initialization
  hwloc_topology_load(topology);  // actual detection
  machine = hwloc_get_root_obj(topology);
  std::hash<std::string> hasher;
  std::string hostname(hwloc_obj_get_info_by_name(machine, "HostName"));
  int id = static_cast<int>(hasher(hostname) % std::numeric_limits<int>::max());
  hwloc_topology_destroy(topology);

  // Make a new communicator based on this hashed hostname
  TIMPI::Communicator new_comm;
  _comm.split(id, _comm.rank(), new_comm);
  // Assign our device index based on the new communicator
  _device_index = new_comm.rank();
  neml_assert(_device_index < _available_devices.size(),
              "MPI split by host would require too many devices");
}

bool
SimpleMPIScheduler::schedule_work_impl(Device & device, std::size_t & batch_size) const
{
  if (_load + _batch_size > _capacity)
    return false;

  device = _available_devices[_device_index];
  batch_size = _batch_size;
  return true;
}

void
SimpleMPIScheduler::dispatched_work_impl(Device, std::size_t n)
{
  _load += n;
}

void
SimpleMPIScheduler::completed_work_impl(Device, std::size_t n)
{
  neml_assert(_load >= n, "Load underflow");
  _load -= n;
}

bool
SimpleMPIScheduler::all_work_completed() const
{
  return _load == 0;
}

} // namespace neml2
