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

#include "neml2/config.h"

#ifdef NEML2_MPI
#include <mpi.h>
#endif

#include "neml2/dispatchers/WorkScheduler.h"
#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"

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
 * The number of devices in the list must be greater than or equal to local communicator size.
 *
 * The list must not include the cpu.
 *
 * \note This scheduler is always registered with the NEML2 Registry so that the syntax
 * catalog reflects its existence. When the library is built without `NEML2_MPI=ON` the
 * constructor throws immediately, instructing the user to rebuild with MPI support.
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

  std::vector<Device> devices() const override { return {_available_devices[_device_index]}; }

#ifdef NEML2_MPI
  virtual MPI_Comm & comm() { return _comm; }

  virtual void set_comm(MPI_Comm comm) { _comm = comm; }
#endif

protected:
  bool schedule_work_impl(Device &, std::size_t &) const override;
  void dispatched_work_impl(Device, std::size_t) override;
  void completed_work_impl(Device, std::size_t) override;
  bool all_work_completed() const override;

private:
#ifdef NEML2_MPI
  /// Figure out which device to use
  void determine_my_device();
#endif

  /// The devices to dispatch to
  std::vector<Device> _available_devices;

  /// The batch size to dispatch
  std::vector<std::size_t> _batch_sizes;

  /// The capacity of the device
  std::vector<std::size_t> _capacities;

#ifdef NEML2_MPI
  /// Global communicator to use to split
  MPI_Comm _comm;
#endif

  /// This rank's device
  std::size_t _device_index = 0;

  /// Current load on the device
  std::size_t _load = 0;
};

#ifdef NEML2_MPI
#define neml2_call_mpi(call)                                                                       \
  do                                                                                               \
  {                                                                                                \
    int err = (call);                                                                              \
    if (err != MPI_SUCCESS)                                                                        \
    {                                                                                              \
      char err_string[MPI_MAX_ERROR_STRING];                                                       \
      int len = 0;                                                                                 \
      MPI_Error_string(err, err_string, &len);                                                     \
      throw NEMLException(std::string("MPI error: ") + err_string);                                \
    }                                                                                              \
  } while (0)
#endif

} // namespace neml2
