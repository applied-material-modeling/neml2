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

  return 0;
}
