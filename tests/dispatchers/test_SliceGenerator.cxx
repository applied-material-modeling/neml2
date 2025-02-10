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

#include "neml2/dispatchers/SliceGenerator.h"
#include "neml2/tensors/indexing.h"

using namespace neml2;

TEST_CASE("SliceGenerator", "[dispatchers]")
{
  SliceGenerator generator(50, 2000);
  REQUIRE(generator.total() == 1950);
  REQUIRE(generator.offset() == 0);

  std::size_t n;
  indexing::Slice work;

  REQUIRE(generator.has_more());
  std::tie(n, work) = generator.next(1);
  REQUIRE(generator.offset() == 1);
  REQUIRE(n == 1);
  REQUIRE(work.start() == 50);
  REQUIRE(work.stop() == 51);

  REQUIRE(generator.has_more());
  std::tie(n, work) = generator.next(2);
  REQUIRE(generator.offset() == 3);
  REQUIRE(n == 2);
  REQUIRE(work.start() == 51);
  REQUIRE(work.stop() == 53);

  REQUIRE(generator.has_more());
  std::tie(n, work) = generator.next(1000);
  REQUIRE(generator.offset() == 1003);
  REQUIRE(n == 1000);
  REQUIRE(work.start() == 53);
  REQUIRE(work.stop() == 1053);

  REQUIRE(generator.has_more());
  std::tie(n, work) = generator.next(946);
  REQUIRE(generator.offset() == 1949);
  REQUIRE(n == 946);
  REQUIRE(work.start() == 1053);
  REQUIRE(work.stop() == 1999);

  REQUIRE(generator.has_more());
  std::tie(n, work) = generator.next(5);
  REQUIRE(generator.offset() == 1950);
  REQUIRE(n == 1);
  REQUIRE(work.start() == 1999);
  REQUIRE(work.stop() == 2000);

  REQUIRE(!generator.has_more());
}
