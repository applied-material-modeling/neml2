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
#include "neml2/csrc/dispatchers/SimpleScheduler.h"

#include "test_util.h"

using namespace neml2::aoti;

int
main()
{
  // SimpleScheduler: cpu, fixed chunk.
  {
    SimpleScheduler s({/*device=*/"cpu", /*batch_size=*/8});
    NEML2_CHECK(s.device().is_cpu());
    NEML2_CHECK(s.batch_size() == 8);
  }

  // SimpleScheduler: a concrete CUDA index parses without needing a GPU present
  // (the device string is parsed, not validated against hardware).
  {
    SimpleScheduler s({/*device=*/"cuda:1", /*batch_size=*/4});
    NEML2_CHECK(s.device().is_cuda());
    NEML2_CHECK(s.device().has_index());
    NEML2_CHECK(s.device().index() == 1);
    NEML2_CHECK(s.batch_size() == 4);
  }

  // batch_size 0 = "no chunking" sentinel, faithfully reported.
  {
    SimpleScheduler s({/*device=*/"cpu", /*batch_size=*/0});
    NEML2_CHECK(s.batch_size() == 0);
  }

  // MPISimpleScheduler always exists, but constructing it outside an MPI runtime must
  // throw -- either "rebuild with -DNEML2_MPI=ON" (flag off) or "MPI has not
  // been initialized" (flag on, no MPI_Init in this plain test process). Either
  // way: a clean exception, no crash.
  {
    MPISimpleScheduler::Config cfg;
    cfg.devices = {"cuda:0"};
    cfg.batch_sizes = {16};
    NEML2_CHECK_THROWS(MPISimpleScheduler{cfg});
  }

  // The comm field defaults to "use MPI_COMM_WORLD" without pulling in any MPI
  // type at the call site.
  {
    MPISimpleScheduler::Config cfg;
    NEML2_CHECK(cfg.comm == nullptr);
  }

  // mpi_device_index: per-node round-robin of the local rank over the device
  // list. MPI-free, so it's tested directly here without an MPI runtime.
  {
    // m > n: 4 ranks, 2 devices -> round-robin 0,1,0,1.
    NEML2_CHECK(mpi_device_index(0, 4, 2) == 0);
    NEML2_CHECK(mpi_device_index(1, 4, 2) == 1);
    NEML2_CHECK(mpi_device_index(2, 4, 2) == 0);
    NEML2_CHECK(mpi_device_index(3, 4, 2) == 1);

    // m == n: one device per rank.
    NEML2_CHECK(mpi_device_index(0, 2, 2) == 0);
    NEML2_CHECK(mpi_device_index(1, 2, 2) == 1);

    // Uneven (3 ranks, 2 devices): both devices used, device 0 serves two ranks.
    NEML2_CHECK(mpi_device_index(0, 3, 2) == 0);
    NEML2_CHECK(mpi_device_index(1, 3, 2) == 1);
    NEML2_CHECK(mpi_device_index(2, 3, 2) == 0);

    // m < n: 1 rank, 2 devices -> idle GPU, which is an error.
    NEML2_CHECK_THROWS(mpi_device_index(0, 1, 2));
  }

  // parse_mpi_devices: CPU and CUDA are both accepted (a pure-CPU MPI run passes
  // {"cpu"}); an empty list or an unsupported device type is rejected. MPI-free,
  // so tested directly without an MPI runtime.
  {
    const auto cpu = parse_mpi_devices({"cpu"});
    NEML2_CHECK(cpu.size() == 1);
    NEML2_CHECK(cpu.front().is_cpu());

    const auto gpus = parse_mpi_devices({"cuda:0", "cuda:1"});
    NEML2_CHECK(gpus.size() == 2);
    NEML2_CHECK(gpus.at(0).is_cuda());
    NEML2_CHECK(gpus.at(1).is_cuda());
    NEML2_CHECK(gpus.at(1).index() == 1);

    const auto mixed = parse_mpi_devices({"cpu", "cuda:0", "cuda:1"});
    NEML2_CHECK(mixed.size() == 3);
    NEML2_CHECK(mixed.at(0).is_cpu());
    NEML2_CHECK(mixed.at(1).is_cuda());
    NEML2_CHECK(mixed.at(2).is_cuda());

    // Bare (unpinned) CUDA devices are allowed when none pins an index.
    const auto bare = parse_mpi_devices({"cuda", "cuda"});
    NEML2_CHECK(bare.size() == 2);

    NEML2_CHECK_THROWS(parse_mpi_devices({}));      // empty list
    NEML2_CHECK_THROWS(parse_mpi_devices({"mps"})); // unsupported device type

    // `cpu` may appear at most once.
    NEML2_CHECK_THROWS(parse_mpi_devices({"cpu", "cpu"}));

    // If any CUDA device pins an index, all must -- and uniquely.
    NEML2_CHECK_THROWS(parse_mpi_devices({"cuda:0", "cuda:0"})); // duplicate index
    NEML2_CHECK_THROWS(parse_mpi_devices({"cuda:0", "cuda"}));   // mixed pinned/unpinned
    NEML2_CHECK_THROWS(parse_mpi_devices({"cuda", "cuda:0"}));   // mixed (other order)
  }

  return 0;
}
