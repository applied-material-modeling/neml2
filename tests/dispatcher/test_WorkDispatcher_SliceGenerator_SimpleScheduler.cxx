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

#include "neml2/dispatcher/SliceGenerator.h"
#include "neml2/dispatcher/SimpleScheduler.h"
#include "neml2/dispatcher/WorkDispatcher.h"
#include "neml2/misc/types.h"

using namespace neml2;

TEST_CASE("WorkDispatcher SliceGenerator SimpleScheduler", "[dispatcher]")
{
  SliceGenerator generator(50, 2000);
  SimpleScheduler scheduler(torch::kCPU, /*batch_size=*/345, /*capacity=*/800);

  SECTION("no reduction, no preprecessing, no postprocessing")
  {
    auto func = [](indexing::Slice && x, torch::Device /*device*/) -> Size
    { return x.start().expect_int() * x.stop().expect_int() * x.step().expect_int(); };

    WorkDispatcher<indexing::Slice, Size> dispatcher(func);

    // The generated slices and results should be
    //   (50, 395, 1) -> 19750
    //   (395, 740, 1) -> 292300
    //   (740, 1085, 1) -> 802900
    //   (1085, 1430, 1) -> 1551550
    //   (1430, 1775, 1) -> 2538250
    //   (1775, 2000, 1) -> 3550000
    std::vector<Size> expected = {19750, 292300, 802900, 1551550, 2538250, 3550000};

    SECTION("run")
    {
      auto result = dispatcher.run(generator, scheduler);
      REQUIRE(result == expected);
    }

    SECTION("run_async")
    {
      auto result = dispatcher.run_async(generator, scheduler);
      REQUIRE(result == expected);
    }
  }

  SECTION("with reduction, no preprecessing, no postprocessing")
  {
    auto func = [](indexing::Slice && x, torch::Device /*device*/) -> Size
    { return x.start().expect_int() * x.stop().expect_int() * x.step().expect_int(); };
    auto red = [](std::vector<Size> && results) -> Size
    {
      Size sum = 0;
      for (const auto & result : results)
        sum += result;
      return sum;
    };

    WorkDispatcher<indexing::Slice, Size, Size> dispatcher(func, red);

    // The generated slices and results should be
    //   (50, 395, 1) -> 19750
    //   (395, 740, 1) -> 292300
    //   (740, 1085, 1) -> 802900
    //   (1085, 1430, 1) -> 1551550
    //   (1430, 1775, 1) -> 2538250
    //   (1775, 2000, 1) -> 3550000
    // The expected result is the sum of the above results

    SECTION("run")
    {
      auto result = dispatcher.run(generator, scheduler);
      REQUIRE(result == 8754750);
    }

    SECTION("run_async")
    {
      auto result = dispatcher.run_async(generator, scheduler);
      REQUIRE(result == 8754750);
    }
  }

  SECTION("with reduction, with preprecessing, with postprocessing")
  {
    auto func = [](indexing::Slice && x, torch::Device /*device*/) -> Size
    { return x.start().expect_int() * x.stop().expect_int() * x.step().expect_int(); };
    auto red = [](std::vector<Size> && results) -> Size
    {
      Size sum = 0;
      for (const auto & result : results)
        sum += result;
      return sum;
    };
    auto preprocess = [](indexing::Slice && x, torch::Device /*device*/) -> indexing::Slice
    { return indexing::Slice(x.start() + 1, x.stop() - 1, x.step()); };
    auto postprocess = [](Size result) -> Size { return result + 1; };

    WorkDispatcher<indexing::Slice, Size, Size> dispatcher(func, red, preprocess, postprocess);

    // The generated slices and results should be
    //   (50, 395, 1) -> (51, 394, 1) -> 20094 -> 20095
    //   (395, 740, 1) -> (396, 739, 1) -> 292644 -> 292645
    //   (740, 1085, 1) -> (741, 1084, 1) -> 803244 -> 803245
    //   (1085, 1430, 1) -> (1086, 1429, 1) -> 1551894 -> 1551895
    //   (1430, 1775, 1) -> (1431, 1774, 1) -> 2538594 -> 2538595
    //   (1775, 2000, 1) -> (1776, 1999, 1) -> 3550224 -> 3550225
    // The expected result is the sum of the above results

    SECTION("run")
    {
      auto result = dispatcher.run(generator, scheduler);
      REQUIRE(result == 8756700);
    }

    SECTION("run_async")
    {
      auto result = dispatcher.run(generator, scheduler);
      REQUIRE(result == 8756700);
    }
  }
}
