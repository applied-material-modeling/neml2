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
#include "neml2/base/EnumSelection.h"

using namespace neml2;

TEST_CASE("EnumSelection", "[base]")
{
  SECTION("Construct from candidate strings")
  {
    EnumSelection es1({"a", "b", "c"}, "b");
    REQUIRE(es1.selection() == "b");
    REQUIRE(es1.value() == 1);

    EnumSelection es2({"a", "bb", "cccc"}, "cccc");
    REQUIRE(es2.selection() == "cccc");
    REQUIRE(es2.value() == 2);
  }

  SECTION("Construct from candidate strings and values")
  {
    EnumSelection es1({"a", "b", "c"}, {5, 2, 1}, "b");
    REQUIRE(es1.selection() == "b");
    REQUIRE(es1.value() == 2);

    EnumSelection es2({"a", "bb", "cccc"}, {-1, -2, -5}, "cccc");
    REQUIRE(es2.selection() == "cccc");
    REQUIRE(es2.value() == -5);

    EnumSelection es3 = es2;
    REQUIRE(es3 == es2);
  }

  SECTION("Modify selection")
  {
    EnumSelection es({"a", "b", "c"}, {5, 2, 1}, "b");
    REQUIRE(es.selection() == "b");
    REQUIRE(es.value() == 2);

    std::stringstream ss("c");
    ss >> es;
    REQUIRE(es.selection() == "c");
    REQUIRE(es.value() == 1);

    std::stringstream ss2("d");
    REQUIRE_THROWS_WITH(ss2 >> es, Catch::Matchers::ContainsSubstring("Invalid selection"));
  }

  SECTION("Test for (in)equality")
  {
    EnumSelection es1({"a", "b", "c"}, {5, 2, 1}, "a");
    EnumSelection es2({"a", "b", "c"}, {5, 2, 1}, "b");
    EnumSelection es3({"a", "b", "c"}, {5, 2, 1}, "c");
    EnumSelection es4({"a", "b", "c"}, {5, 2, 3}, "c");
    EnumSelection es5({"a", "d", "c"}, {5, 2, 1}, "c");
    EnumSelection es6({"b", "a", "c"}, {2, 5, 1}, "a");

    REQUIRE(es1 != es2);
    REQUIRE(es1 != es3);
    REQUIRE(es2 != es3);
    REQUIRE(es3 != es4);
    REQUIRE(es3 != es5);
    REQUIRE(es1 == es6);
  }

  SECTION("Cast to enum class")
  {
    enum class SomeEnum : int
    {
      a = 5,
      b = 2,
      c = 1
    };

    EnumSelection es1({"a", "b", "c"}, {5, 2, 1}, "a");
    auto value = es1.as<SomeEnum>();
    REQUIRE(value == SomeEnum::a);
  }

  SECTION("stringify")
  {
    enum class SomeEnum : int
    {
      a = 5,
      b = 2,
      c = 1
    };

    EnumSelection es1({"a", "b", "c"}, {5, 2, 1}, "a");
    REQUIRE(utils::stringify(es1) == "a");
  }

  SECTION("Errors")
  {
    REQUIRE_THROWS_WITH(
        EnumSelection({"a", "a", "b"}, "a"),
        Catch::Matchers::ContainsSubstring("Choices of (Multi)EnumSelection must be unique"));
    REQUIRE_THROWS_WITH(EnumSelection({"a", "b", "c"}, "d"),
                        Catch::Matchers::ContainsSubstring("Invalid selection for EnumSelection"));
    REQUIRE_THROWS_WITH(EnumSelection({"a", "b", "c"}, {1, 2, 3}, "d"),
                        Catch::Matchers::ContainsSubstring("Invalid selection for EnumSelection"));
    REQUIRE_THROWS_WITH(
        EnumSelection({"a", "b", "c"}, {2, 2}, "a"),
        Catch::Matchers::ContainsSubstring("number of choices must match the number of values"));
    REQUIRE_THROWS_WITH(
        EnumSelection({"a", "b", "c"}, {2, 2, 3}, "a"),
        Catch::Matchers::ContainsSubstring("Values of (Multi)EnumSelection must be unique"));
  }
}
