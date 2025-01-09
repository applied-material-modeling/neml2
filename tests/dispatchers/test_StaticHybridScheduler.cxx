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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/dispatchers/StaticHybridScheduler.h"

using namespace neml2;

TEST_CASE("StaticHybridScheduler", "[dispatchers]")
{
  SECTION("single device")
  {
    StaticHybridScheduler scheduler(
        /*device_list=*/{Device("cpu")},
        /*batch_sizes=*/{2},
        /*capacities=*/{3},
        /*priorities=*/{1});
    const auto & status = scheduler.status();
    REQUIRE(status.size() == 1);
    REQUIRE(status[0].device == Device("cpu"));
    REQUIRE(status[0].batch_size == 2);
    REQUIRE(status[0].capacity == 3);
    REQUIRE(status[0].priority == Catch::Approx(1.0));
    REQUIRE(status[0].load == 0);

    scheduler.dispatched_work(kCPU, 1);
    REQUIRE(status[0].load == 1);

    scheduler.dispatched_work(kCPU, 2);
    REQUIRE(status[0].load == 3);

    scheduler.completed_work(kCPU, 1);
    REQUIRE(status[0].load == 2);
  }

  SECTION("multiple devices")
  {
    StaticHybridScheduler scheduler(
        /*device_list=*/{Device("cpu"), Device("cuda:0"), Device("cuda:1")},
        /*batch_sizes=*/{2, 3, 4},
        /*capacities=*/{3, 4, 5},
        /*priorities=*/{3, 2, 1});
    const auto & status = scheduler.status();
    REQUIRE(status.size() == 3);

    REQUIRE(status[0].device == Device("cpu"));
    REQUIRE(status[0].batch_size == 2);
    REQUIRE(status[0].capacity == 3);
    REQUIRE(status[0].priority == Catch::Approx(3.0));
    REQUIRE(status[0].load == 0);

    REQUIRE(status[1].device == Device("cuda:0"));
    REQUIRE(status[1].batch_size == 3);
    REQUIRE(status[1].capacity == 4);
    REQUIRE(status[1].priority == Catch::Approx(2.0));
    REQUIRE(status[1].load == 0);

    REQUIRE(status[2].device == Device("cuda:1"));
    REQUIRE(status[2].batch_size == 4);
    REQUIRE(status[2].capacity == 5);
    REQUIRE(status[2].priority == Catch::Approx(1.0));
    REQUIRE(status[2].load == 0);

    scheduler.dispatched_work(Device("cpu"), 1);
    REQUIRE(status[0].load == 1);

    scheduler.dispatched_work(Device("cuda:0"), 2);
    REQUIRE(status[1].load == 2);

    scheduler.dispatched_work(Device("cuda:1"), 3);
    REQUIRE(status[2].load == 3);

    scheduler.completed_work(Device("cpu"), 1);
    REQUIRE(status[0].load == 0);

    scheduler.completed_work(Device("cuda:0"), 2);
    REQUIRE(status[1].load == 0);

    scheduler.completed_work(Device("cuda:1"), 3);
    REQUIRE(status[2].load == 0);
  }
}
