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

#include "neml2/misc/string_utils.h"
#include "neml2/base/LabeledAxisAccessor.h"

using namespace neml2;

TEST_CASE("LabeledAxisAccessor", "[models]")
{
  LabeledAxisAccessor a("a", "b", "c");

  SECTION("struct LabeledAxisAccessor")
  {
    SECTION("LabeledAxisAccessor")
    {
      REQUIRE_THROWS_WITH(LabeledAxisAccessor("a.b", "c", "d"),
                          Catch::Matchers::ContainsSubstring("Invalid item name"));
    }

    SECTION("vec") { REQUIRE(a.vec() == std::vector<std::string>{"a", "b", "c"}); }

    SECTION("empty")
    {
      REQUIRE(!a.empty());
      REQUIRE(a.slice(3).empty());
    }

    SECTION("operator std::vector<std::string>()")
    {
      REQUIRE(std::vector<std::string>(a) == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("with_suffix")
    {
      REQUIRE(a.with_suffix("s").vec() == std::vector<std::string>{"a", "b", "cs"});
    }

    SECTION("append")
    {
      REQUIRE(a.append("d").vec() == std::vector<std::string>{"a", "b", "c", "d"});
    }

    SECTION("on")
    {
      REQUIRE(a.prepend("x").vec() == std::vector<std::string>{"x", "a", "b", "c"});
      LabeledAxisAccessor b("d", "e", "f");
      REQUIRE(a.prepend(b).vec() == std::vector<std::string>{"d", "e", "f", "a", "b", "c"});
    }

    SECTION("slice")
    {
      REQUIRE(a.slice(1).vec() == std::vector<std::string>{"b", "c"});
      REQUIRE(a.slice(1, 3).vec() == std::vector<std::string>{"b", "c"});
      REQUIRE(a.slice(0, 2).vec() == std::vector<std::string>{"a", "b"});
    }
  }

  SECTION("operator==")
  {
    auto b = a;
    REQUIRE(a == b);
  }

  SECTION("operator!=")
  {
    auto b = a;
    REQUIRE(a != b.slice(1));
  }

  SECTION("operator<<") { REQUIRE(utils::stringify(a) == "a/b/c"); }
}
