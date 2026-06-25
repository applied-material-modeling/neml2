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
#include <string>
#include <vector>

#include "neml2/csrc/dispatchers/WorkScheduler.h"

namespace neml2::aoti
{
/**
 * @brief Assign devices to MPI ranks, round-robin within each node.
 *
 * For MPI jobs that drive several devices from many ranks: each rank picks a
 * device from `devices` (CPU or CUDA) based on its rank *within its node* (ranks
 * are grouped by hostname, then `local_rank % devices.size()` indexes into the
 * list). With `m` ranks on a node and `n = devices.size()`: `m == n` gives one
 * device per rank, `m > n` wraps round-robin (so a device may serve several
 * ranks), and `m < n` is an error -- idle devices are disallowed. Within a rank
 * the workload is then chunked exactly like @ref SimpleScheduler. A pure-CPU run
 * typically passes a single `{"cpu"}` so every rank chunks on the CPU.
 *
 * The per-node split is **collective** over the chosen communicator, so every
 * rank in it must construct the scheduler.
 *
 * Requires NEML2 built with `-DNEML2_MPI=ON`; otherwise the constructor throws.
 * The header is deliberately free of any MPI types so it is includable (and the
 * class layout is identical) regardless of the build flag -- all MPI machinery
 * lives in the translation unit. The host application owns `MPI_Init`/
 * `MPI_Finalize`; the communicator defaults to `MPI_COMM_WORLD` but a
 * subcommunicator can be supplied via `Config::comm`.
 */
class AOTI_EXPORT MPISimpleScheduler : public SyncScheduler
{
public:
  struct Config
  {
    /// Devices to choose from (CPU or CUDA), e.g. {"cuda:0", "cuda:1"} or
    /// {"cpu"}. Each entry must denote a distinct device: `cpu` and unpinned
    /// `cuda` may each appear at most once, pinned CUDA indices must be unique,
    /// and bare `cuda` cannot be mixed with `cuda:N`. At least one rank per
    /// device on each node is required (idle devices are an error).
    std::vector<std::string> devices;
    /// Per-device chunk size. Length 1 broadcasts to all devices; otherwise it
    /// must match `devices`.
    std::vector<std::size_t> batch_sizes;
    /// Communicator to assign devices over. nullptr => MPI_COMM_WORLD; otherwise
    /// points at the caller's MPI_Comm (read during construction only), e.g.
    /// `cfg.comm = &my_subcomm;`. Typed as void* so this header stays MPI-free.
    const void * comm = nullptr;
  };

  explicit MPISimpleScheduler(const Config & config);

  at::Device device() const override { return _devices.at(_device_index); }
  std::size_t batch_size() const override
  {
    return _batch_sizes.size() == 1 ? _batch_sizes.front() : _batch_sizes.at(_device_index);
  }

private:
  std::vector<at::Device> _devices;
  std::vector<std::size_t> _batch_sizes;
  std::size_t _device_index = 0;
};

/// Per-node device assignment: round-robin the local rank over the device list.
/// Requires `local_size >= ndevices` (idle GPUs disallowed) -- throws otherwise.
/// Free-standing and MPI-free (compiled in every build config) so it is exposed
/// for direct unit testing without an MPI runtime.
AOTI_EXPORT std::size_t
mpi_device_index(std::size_t local_rank, std::size_t local_size, std::size_t ndevices);

/// Parse + validate the scheduler's device list: it must be non-empty, every
/// entry must name a CPU or CUDA device (the only devices the dispatcher
/// supports), and each entry must denote a distinct device -- `cpu` and unpinned
/// `cuda` each at most once, pinned CUDA indices unique, and no mixing bare
/// `cuda` with `cuda:N`. Free-standing and MPI-free (compiled in every build
/// config) so it is exposed for direct unit testing without an MPI runtime.
AOTI_EXPORT std::vector<at::Device> parse_mpi_devices(const std::vector<std::string> & devices);
} // namespace neml2::aoti
