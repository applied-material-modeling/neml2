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

  options.set<std::vector<std::size_t>>("batch_sizes");
  options.set("batch_sizes").doc() = "List of batch sizes for each device";

  options.set<std::vector<std::size_t>>("capacities") = {};
  options.set("capacities").doc() = "List of capacities for each device, default to batch_sizes";

  return options;
}

SimpleMPIScheduler::SimpleMPIScheduler(const OptionSet & options)
  : WorkScheduler(options),
    _available_devices(options.get<std::vector<Device>>("devices")),
    _batch_sizes(options.get<std::vector<std::size_t>>("batch_sizes")),
    _capacities(options.get("capacities").user_specified()
                    ? options.get<std::vector<std::size_t>>("capacities")
                    : _batch_sizes),
    _comm(TIMPI::Communicator(MPI_COMM_WORLD))
{
  neml_assert(_available_devices.size() == _batch_sizes.size(),
              "Number of batch sizes must match the number of devices.");
  neml_assert(
      _available_devices.size() == _capacities.size(),
      "Number of capacities must match the number of devices and the number of batch sizes.");
}

void
SimpleMPIScheduler::setup()
{
  WorkScheduler::setup();

  // First pass:
  // - Prohibit any CPU
  // - Check if any CUDA device is present
  bool cuda = false;
  for (const auto & device : _available_devices)
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
  bool has_multiple_cuda_devices = _available_devices.size() > 1 && cuda;
  if (has_multiple_cuda_devices)
  {
    std::set<DeviceIndex> cuda_device_ids;
    for (const auto & device : _available_devices)
    {
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
  // NOLINTBEGIN(modernize-avoid-c-arrays)
  char c_str_hostname[MPI_MAX_PROCESSOR_NAME];
  int name_len = 0;
  timpi_call_mpi(MPI_Get_processor_name(c_str_hostname, &name_len));
  std::string hostname = std::string(c_str_hostname);

  std::hash<std::string> hasher;
  int id = static_cast<int>(hasher(hostname) % std::numeric_limits<int>::max());

  // Make a new communicator based on this hashed hostname
  TIMPI::Communicator new_comm;
  _comm.split(id, int(_comm.rank()), new_comm);
  // Assign our device index based on the new communicator
  _device_index = new_comm.rank();
  neml_assert(new_comm.size() <= _available_devices.size(),
              "MPI split by host would require too many devices");
  // NOLINTEND(modernize-avoid-c-arrays)
}

bool
SimpleMPIScheduler::schedule_work_impl(Device & device, std::size_t & batch_size) const
{
  if (_load + _batch_sizes[_device_index] > _capacities[_device_index])
    return false;

  device = _available_devices[_device_index];
  batch_size = _batch_sizes[_device_index];
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
