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
#include "neml2/dispatcher/UniformSequentialDispatcher.h"
#include "neml2/misc/types.h"

using namespace neml2;

TEST_CASE("UniformSequentialDispatcher", "[dispatcher]")
{
  // A dummy function to test the dispatcher
  auto func = [](indexing::Slice && x) -> Size
  { return x.start().expect_int() * x.stop().expect_int() * x.step().expect_int(); };

  // A dummy reduction function
  auto reduction = [](std::vector<Size> && results) -> Size
  {
    Size sum = 0;
    for (const auto & result : results)
      sum += result;
    return sum;
  };

  UniformSequentialDispatcher<indexing::Slice, Size, Size> dispatcher(345, func, reduction);
  SliceGenerator generator(50, 2000);
  auto result = dispatcher.run(generator);

  // The generated slices and results should be
  //   (50, 395, 1) -> 19750
  //   (395, 740, 1) -> 292300
  //   (740, 1085, 1) -> 802900
  //   (1085, 1430, 1) -> 1551550
  //   (1430, 1775, 1) -> 2538250
  //   (1775, 2000, 1) -> 3550000
  // After reduction, the result should be
  //   8754750
  REQUIRE(result == 8754750);
}
