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

// Unit test for the StaticHybridScheduler greedy load-tracking policy. Pure
// scheduler logic -- no Model, no GPU (device strings are only parsed). The
// schedule_work calls below always have a device available, so they never
// block this single thread.

#include <cstddef>
#include <memory>

#include <c10/core/Device.h>

#include "neml2/csrc/dispatchers/StaticHybridScheduler.h"

#include "test_util.h"

using namespace neml2::aoti;

int
main()
{
  // Greedy: fill the highest-priority device that still has spare capacity.
  {
    StaticHybridScheduler::Config cfg;
    cfg.devices = {"cpu", "cuda:0"};
    cfg.batch_sizes = {4, 4};
    cfg.capacities = {4, 8}; // cpu holds 1 chunk, cuda holds 2
    cfg.priorities = {1.0, 2.0};
    StaticHybridScheduler s(cfg);
    NEML2_CHECK(s.devices().size() == 2);

    at::Device dev = at::kCPU;
    std::size_t n = 0;

    // #1: both free -> cuda:0 wins on priority.
    s.schedule_work(dev, n);
    NEML2_CHECK(dev.is_cuda() && n == 4);
    s.dispatched_work(dev, n); // cuda load 4 (<= 8, still has room)

    // #2: cuda still has room (4+4<=8) and higher priority -> cuda again.
    s.schedule_work(dev, n);
    NEML2_CHECK(dev.is_cuda());
    s.dispatched_work(dev, n); // cuda load 8 -> full

    // #3: cuda full -> falls to cpu.
    s.schedule_work(dev, n);
    NEML2_CHECK(dev.is_cpu() && n == 4);
    s.dispatched_work(dev, n); // cpu load 4 -> full

    // Both full now; draining frees them and wait_for_completion returns.
    s.completed_work(at::Device("cuda:0"), 8);
    s.completed_work(at::Device("cpu"), 4);
    s.wait_for_completion();

    // After draining, the highest-priority device is offered again.
    s.schedule_work(dev, n);
    NEML2_CHECK(dev.is_cuda());
  }

  // Config validation.
  auto build = [](StaticHybridScheduler::Config c)
  { return std::make_shared<StaticHybridScheduler>(c); };

  { // empty devices
    StaticHybridScheduler::Config c;
    c.batch_sizes = {1};
    NEML2_CHECK_THROWS(build(c));
  }
  { // duplicate device
    StaticHybridScheduler::Config c;
    c.devices = {"cuda:0", "cuda:0"};
    c.batch_sizes = {1};
    NEML2_CHECK_THROWS(build(c));
  }
  { // more than one CPU (intra-op oversubscription) is rejected
    StaticHybridScheduler::Config c;
    c.devices = {"cpu", "cpu:0"};
    c.batch_sizes = {1};
    NEML2_CHECK_THROWS(build(c));
  }
  { // capacity < batch_size (a chunk could never be placed)
    StaticHybridScheduler::Config c;
    c.devices = {"cuda:0"};
    c.batch_sizes = {8};
    c.capacities = {4};
    NEML2_CHECK_THROWS(build(c));
  }
  { // batch_size 0
    StaticHybridScheduler::Config c;
    c.devices = {"cuda:0"};
    c.batch_sizes = {0};
    NEML2_CHECK_THROWS(build(c));
  }
  { // batch_sizes length mismatch
    StaticHybridScheduler::Config c;
    c.devices = {"cpu", "cuda:0"};
    c.batch_sizes = {1, 2, 3};
    NEML2_CHECK_THROWS(build(c));
  }

  // Broadcast forms: single batch_size / capacity / priority applied to all.
  {
    StaticHybridScheduler::Config c;
    c.devices = {"cpu", "cuda:0"};
    c.batch_sizes = {8}; // broadcast
    StaticHybridScheduler s(c);
    NEML2_CHECK(s.devices().size() == 2);
  }

  return 0;
}
