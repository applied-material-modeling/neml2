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
 * @brief Assign one device per MPI rank.
 *
 * For larger MPI jobs that run one rank per accelerator: each rank picks a
 * device from `devices` based on its rank *within its node* (ranks are grouped
 * by hostname, then the local rank indexes into the device list). Within a rank
 * the workload is then chunked exactly like @ref SimpleScheduler.
 *
 * Requires NEML2 built with `-DNEML2_MPI=ON`; otherwise the constructor throws.
 * The header is deliberately free of any MPI types so it is includable (and the
 * class layout is identical) regardless of the build flag -- all MPI machinery
 * lives in the translation unit. The host application owns `MPI_Init`/
 * `MPI_Finalize`; the communicator used is `MPI_COMM_WORLD`.
 */
class AOTI_EXPORT MPISimpleScheduler : public SyncScheduler
{
public:
  struct Config
  {
    /// CUDA devices to choose from, e.g. {"cuda:0", "cuda:1"}. One device per
    /// rank per node is required.
    std::vector<std::string> devices;
    /// Per-device chunk size. Length 1 broadcasts to all devices; otherwise it
    /// must match `devices`.
    std::vector<std::size_t> batch_sizes;
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
} // namespace neml2::aoti
