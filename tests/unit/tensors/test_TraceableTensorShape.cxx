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

#include "neml2/tensors/TraceableTensorShape.h"

using namespace neml2;

TEST_CASE("TraceableTensorShape", "[tensors]")
{
  SECTION("construction from TensorShape")
  {
    TensorShape shape{2, 3, 4};
    TraceableTensorShape tts(shape);
    REQUIRE(tts.size() == 3);
    REQUIRE(tts[0].concrete() == 2);
    REQUIRE(tts[1].concrete() == 3);
    REQUIRE(tts[2].concrete() == 4);
  }

  SECTION("construction from TensorShapeRef")
  {
    TensorShapeRef ref({5, 7});
    TraceableTensorShape tts(ref);
    REQUIRE(tts.size() == 2);
    REQUIRE(tts[0].concrete() == 5);
    REQUIRE(tts[1].concrete() == 7);
  }

  SECTION("construction from single Size")
  {
    TraceableTensorShape tts(Size{8});
    REQUIRE(tts.size() == 1);
    REQUIRE(tts[0].concrete() == 8);
  }

  SECTION("construction from ATensor (1-D int64)")
  {
    auto raw = at::tensor({int64_t{2}, int64_t{3}, int64_t{5}});
    TraceableTensorShape tts(raw);
    REQUIRE(tts.size() == 3);
    REQUIRE(tts[0].concrete() == 2);
    REQUIRE(tts[1].concrete() == 3);
    REQUIRE(tts[2].concrete() == 5);
  }

  SECTION("slice(N, M)")
  {
    TraceableTensorShape tts(TensorShape{1, 2, 3, 4, 5});

    SECTION("middle slice")
    {
      auto s = tts.slice(1, 3);
      REQUIRE(s.size() == 3);
      REQUIRE(s[0].concrete() == 2);
      REQUIRE(s[1].concrete() == 3);
      REQUIRE(s[2].concrete() == 4);
    }

    SECTION("from start")
    {
      auto s = tts.slice(0, 2);
      REQUIRE(s.size() == 2);
      REQUIRE(s[0].concrete() == 1);
      REQUIRE(s[1].concrete() == 2);
    }
  }

  SECTION("slice(N) — drop first N")
  {
    TraceableTensorShape tts(TensorShape{10, 20, 30, 40});

    SECTION("drop first 1")
    {
      auto s = tts.slice(1);
      REQUIRE(s.size() == 3);
      REQUIRE(s[0].concrete() == 20);
    }

    SECTION("drop all")
    {
      auto s = tts.slice(4);
      REQUIRE(s.empty());
    }

    SECTION("drop none")
    {
      auto s = tts.slice(0);
      REQUIRE(s.size() == 4);
    }
  }

  SECTION("concrete")
  {
    TraceableTensorShape tts(TensorShape{3, 6, 9});
    const auto c = tts.concrete();
    REQUIRE(c.size() == 3);
    REQUIRE(c[0] == 3);
    REQUIRE(c[1] == 6);
    REQUIRE(c[2] == 9);
  }

  SECTION("as_tensor")
  {
    TraceableTensorShape tts(TensorShape{4, 5});
    const auto t = tts.as_tensor();
    REQUIRE(t.dim() == 1);
    REQUIRE(t.size(0) == 2);
    REQUIRE(t[0].item<Size>() == 4);
    REQUIRE(t[1].item<Size>() == 5);
  }

  SECTION("as_tensor on empty shape returns undefined tensor")
  {
    TraceableTensorShape tts(TensorShape{});
    const auto t = tts.as_tensor();
    REQUIRE(!t.defined());
  }

  SECTION("comparison operators")
  {
    TraceableTensorShape a(TensorShape{2, 3});
    TraceableTensorShape b(TensorShape{2, 3});
    TraceableTensorShape c(TensorShape{2, 4});

    REQUIRE(a == b);
    REQUIRE_FALSE(a != b);
    REQUIRE_FALSE(a == c);
    REQUIRE(a != c);
  }
}
