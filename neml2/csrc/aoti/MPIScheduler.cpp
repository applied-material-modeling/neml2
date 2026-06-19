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

#include "neml2/csrc/aoti/MPIScheduler.h"
#include "neml2/csrc/aoti/assertions.h"

#ifdef NEML2_MPI
#include <functional>
#include <limits>

#include <mpi.h>
#endif

namespace neml2::aoti
{
#ifndef NEML2_MPI

MPIScheduler::MPIScheduler(const Config & /*config*/)
{
  _assert(false,
          "MPIScheduler requires NEML2 to be built with -DNEML2_MPI=ON. "
          "Rebuild with MPI support to use this scheduler.");
}

#else // NEML2_MPI

namespace
{
// Pick this rank's device index: group the world's ranks by hostname (so ranks
// sharing a node form one group), then use the rank *within that group* as the
// index into the device list. Mirrors the v2 SimpleMPIScheduler assignment.
std::size_t
determine_device_index(std::size_t ndevices)
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int len = 0;
  MPI_Get_processor_name(name, &len);

  // Hash the hostname to a non-negative MPI "color"; ranks with equal color
  // (same host) land in the same split communicator.
  const std::size_t h = std::hash<std::string>{}(std::string(name, static_cast<std::size_t>(len)));
  const int color = static_cast<int>(h % static_cast<std::size_t>(std::numeric_limits<int>::max()));

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Comm local = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local);
  int local_rank = 0;
  MPI_Comm_rank(local, &local_rank);
  MPI_Comm_free(&local);

  const auto idx = static_cast<std::size_t>(local_rank);
  _assert(idx < ndevices,
          "MPIScheduler: local MPI rank ",
          idx,
          " on this node exceeds the ",
          ndevices,
          " device(s) provided. Provide at least one device per rank on each node.");
  return idx;
}
} // namespace

MPIScheduler::MPIScheduler(const Config & config)
  : _batch_sizes(config.batch_sizes)
{
  _assert(!config.devices.empty(), "MPIScheduler: `devices` must be non-empty.");
  _assert(!_batch_sizes.empty(), "MPIScheduler: `batch_sizes` must be non-empty.");
  _assert(_batch_sizes.size() == 1 || _batch_sizes.size() == config.devices.size(),
          "MPIScheduler: `batch_sizes` must have length 1 (broadcast) or match `devices` (",
          config.devices.size(),
          "); got ",
          _batch_sizes.size(),
          ".");

  _devices.reserve(config.devices.size());
  for (const auto & d : config.devices)
  {
    at::Device dev(d);
    _assert(dev.is_cuda(),
            "MPIScheduler: device '",
            d,
            "' is not a CUDA device. The MPI scheduler assigns one GPU per rank; "
            "use SimpleScheduler for CPU dispatch.");
    _devices.push_back(dev);
  }

  // The host (mpirun harness / MOOSE app) owns MPI's lifetime.
  int inited = 0;
  MPI_Initialized(&inited);
  _assert(inited,
          "MPIScheduler: MPI has not been initialized. The host application must call "
          "MPI_Init before constructing the scheduler.");

  _device_index = determine_device_index(_devices.size());
}

#endif // NEML2_MPI
} // namespace neml2::aoti
