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

#include <sstream>
#include <catch2/catch_test_macros.hpp>

#include "neml2/misc/defaults.h"
#include "neml2/tensors/TraceableSize.h"
#include "neml2/tensors/Scalar.h"

using namespace neml2;

TEST_CASE("TraceableSize", "[tensors]")
{
  SECTION("concrete path (holds Size)")
  {
    TraceableSize ts(Size{5});

    SECTION("traceable returns nullptr") { REQUIRE(ts.traceable() == nullptr); }

    SECTION("concrete returns the stored value") { REQUIRE(ts.concrete() == 5); }

    SECTION("as_tensor returns a scalar tensor with the correct value")
    {
      const auto t = ts.as_tensor();
      REQUIRE(t.item<Size>() == 5);
    }

    SECTION("as_scalar returns a Scalar with the correct value")
    {
      const auto s = ts.as_scalar();
      REQUIRE(s.item<double>() == Catch::Approx(5.0));
    }
  }

  SECTION("traceable path (holds ATensor)")
  {
    const auto raw = at::tensor(int64_t{7});
    TraceableSize ts(raw);

    SECTION("traceable returns non-null pointer") { REQUIRE(ts.traceable() != nullptr); }

    SECTION("concrete extracts value from tensor") { REQUIRE(ts.concrete() == 7); }

    SECTION("as_tensor returns the stored tensor")
    {
      const auto t = ts.as_tensor();
      REQUIRE(t.item<Size>() == 7);
    }
  }

  SECTION("operator*")
  {
    SECTION("concrete * concrete")
    {
      TraceableSize a(Size{3});
      TraceableSize b(Size{4});
      const auto c = a * b;
      REQUIRE(c.concrete() == 12);
      REQUIRE(c.traceable() == nullptr);
    }

    SECTION("traceable * concrete")
    {
      const auto raw = at::tensor(int64_t{3});
      TraceableSize a(raw);
      TraceableSize b(Size{4});
      const auto c = a * b;
      REQUIRE(c.concrete() == 12);
      REQUIRE(c.traceable() != nullptr);
    }

    SECTION("concrete * traceable")
    {
      TraceableSize a(Size{3});
      const auto raw = at::tensor(int64_t{4});
      TraceableSize b(raw);
      const auto c = a * b;
      REQUIRE(c.concrete() == 12);
      REQUIRE(c.traceable() != nullptr);
    }

    SECTION("traceable * traceable")
    {
      const auto ra = at::tensor(int64_t{3});
      const auto rb = at::tensor(int64_t{4});
      TraceableSize a(ra);
      TraceableSize b(rb);
      const auto c = a * b;
      REQUIRE(c.concrete() == 12);
      REQUIRE(c.traceable() != nullptr);
    }
  }

  SECTION("comparison operators")
  {
    SECTION("equal concrete sizes")
    {
      TraceableSize a(Size{6});
      TraceableSize b(Size{6});
      REQUIRE(a == b);
      REQUIRE_FALSE(a != b);
    }

    SECTION("unequal concrete sizes")
    {
      TraceableSize a(Size{6});
      TraceableSize b(Size{7});
      REQUIRE_FALSE(a == b);
      REQUIRE(a != b);
    }

    SECTION("traceable equal concrete equivalent")
    {
      const auto raw = at::tensor(int64_t{6});
      TraceableSize a(raw);
      TraceableSize b(Size{6});
      REQUIRE(a == b);
    }
  }

  SECTION("streaming operator")
  {
    TraceableSize ts(Size{42});
    std::ostringstream oss;
    oss << ts;
    REQUIRE(oss.str() == "42");
  }
}
