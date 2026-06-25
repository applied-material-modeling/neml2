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

#include "neml2/csrc/dispatchers/MPISimpleScheduler.h"
#include "neml2/csrc/aoti/assertions.h"

#ifdef NEML2_MPI
#include <functional>
#include <limits>

#include <mpi.h>
#endif

namespace neml2::aoti
{
// Defined unconditionally (MPI-free) so the test suite can link it without an MPI
// runtime. The MPI rank/size discovery under NEML2_MPI funnels into it.
std::size_t
mpi_device_index(std::size_t local_rank, std::size_t local_size, std::size_t ndevices)
{
  _assert(local_size >= ndevices,
          "MPISimpleScheduler: this node has ",
          local_size,
          " rank(s) but ",
          ndevices,
          " device(s) were provided. Idle GPUs are not allowed; launch at least "
          "one rank per device, or pass fewer devices.");
  // ndevices >= 1 is guaranteed by the constructor's non-empty-devices assert.
  return local_rank % ndevices;
}

#ifndef NEML2_MPI

MPISimpleScheduler::MPISimpleScheduler(const Config & /*config*/)
{
  _assert(false,
          "MPISimpleScheduler requires NEML2 to be built with -DNEML2_MPI=ON. "
          "Rebuild with MPI support to use this scheduler.");
}

#else // NEML2_MPI

namespace
{
// Group `comm`'s ranks by hostname (so ranks sharing a node form one group),
// then round-robin the local rank over the device list via mpi_device_index.
std::size_t
determine_device_index(MPI_Comm comm, std::size_t ndevices)
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int len = 0;
  MPI_Get_processor_name(name, &len);

  // Hash the hostname to a non-negative MPI "color"; ranks with equal color
  // (same host) land in the same split communicator.
  const std::size_t h = std::hash<std::string>{}(std::string(name, static_cast<std::size_t>(len)));
  const int color = static_cast<int>(h % static_cast<std::size_t>(std::numeric_limits<int>::max()));

  int world_rank = 0;
  MPI_Comm_rank(comm, &world_rank);

  // Collective over `comm`: every rank in it must reach this point.
  MPI_Comm local = MPI_COMM_NULL;
  MPI_Comm_split(comm, color, world_rank, &local);
  int local_rank = 0;
  int local_size = 0;
  MPI_Comm_rank(local, &local_rank);
  MPI_Comm_size(local, &local_size); // read before freeing the local comm
  MPI_Comm_free(&local);

  return mpi_device_index(
      static_cast<std::size_t>(local_rank), static_cast<std::size_t>(local_size), ndevices);
}
} // namespace

MPISimpleScheduler::MPISimpleScheduler(const Config & config)
  : _batch_sizes(config.batch_sizes)
{
  _assert(!config.devices.empty(), "MPISimpleScheduler: `devices` must be non-empty.");
  _assert(!_batch_sizes.empty(), "MPISimpleScheduler: `batch_sizes` must be non-empty.");
  _assert(_batch_sizes.size() == 1 || _batch_sizes.size() == config.devices.size(),
          "MPISimpleScheduler: `batch_sizes` must have length 1 (broadcast) or match `devices` (",
          config.devices.size(),
          "); got ",
          _batch_sizes.size(),
          ".");

  _devices.reserve(config.devices.size());
  for (const auto & d : config.devices)
  {
    at::Device dev(d);
    _assert(dev.is_cuda(),
            "MPISimpleScheduler: device '",
            d,
            "' is not a CUDA device. The MPI scheduler assigns CUDA devices to "
            "ranks; use SimpleScheduler for CPU dispatch.");
    _devices.push_back(dev);
  }

  // The host (mpirun harness / MOOSE app) owns MPI's lifetime.
  int inited = 0;
  MPI_Initialized(&inited);
  _assert(inited,
          "MPISimpleScheduler: MPI has not been initialized. The host application must call "
          "MPI_Init before constructing the scheduler.");

  // Copy the handle out of the caller's MPI_Comm (sound for both an int handle
  // and an opaque pointer); nullptr selects the world communicator.
  MPI_Comm comm = config.comm ? *static_cast<const MPI_Comm *>(config.comm) : MPI_COMM_WORLD;
  _device_index = determine_device_index(comm, _devices.size());
}

#endif // NEML2_MPI
} // namespace neml2::aoti
