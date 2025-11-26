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
#include "neml2/base/MultiEnumSelection.h"

using namespace neml2;

TEST_CASE("MultiEnumSelection", "[base]")
{
  SECTION("Construct from candidate strings")
  {
    MultiEnumSelection es1({"a", "b", "c"}, {"b"});
    REQUIRE(es1.selections() == std::vector<std::string>{"b"});
    REQUIRE(es1.values() == std::vector<int>{1});

    MultiEnumSelection es2({"a", "bb", "cccc"}, {"cccc", "a"});
    REQUIRE(es2.selections() == std::vector<std::string>{"cccc", "a"});
    REQUIRE(es2.values() == std::vector<int>{2, 0});
  }

  SECTION("Construct from candidate strings and values")
  {
    MultiEnumSelection es1({"a", "b", "c"}, {5, 2, 1}, {"b"});
    REQUIRE(es1.selections() == std::vector<std::string>{"b"});
    REQUIRE(es1.values() == std::vector<int>{2});

    MultiEnumSelection es2({"a", "bb", "cccc"}, {-1, -2, -5}, {"cccc", "a"});
    REQUIRE(es2.selections() == std::vector<std::string>{"cccc", "a"});
    REQUIRE(es2.values() == std::vector<int>{-5, -1});

    MultiEnumSelection es3 = es2;
    REQUIRE(es3 == es2);
  }

  SECTION("Modify selection")
  {
    MultiEnumSelection es({"a", "b", "c"}, {5, 2, 1}, {"b"});
    REQUIRE(es.selections() == std::vector<std::string>{"b"});
    REQUIRE(es.values() == std::vector<int>{2});

    std::stringstream ss("c a");
    ss >> es;
    REQUIRE(es.selections() == std::vector<std::string>{"c", "a"});
    REQUIRE(es.values() == std::vector<int>{1, 5});

    std::stringstream ss2("c d");
    REQUIRE_THROWS_WITH(ss2 >> es, Catch::Matchers::ContainsSubstring("Invalid selection"));
  }

  SECTION("Test for (in)equality")
  {
    MultiEnumSelection es1({"a", "b", "c"}, {5, 2, 1}, {"a"});
    MultiEnumSelection es2({"a", "b", "c"}, {5, 2, 1}, {"b", "a"});
    MultiEnumSelection es3({"a", "b", "c"}, {5, 2, 1}, {"a", "b"});

    REQUIRE(es1 != es2);
    REQUIRE(es1 != es3);
    REQUIRE(es2 != es3);
  }

  SECTION("Cast to enum class")
  {
    enum class SomeEnum : int
    {
      a = 5,
      b = 2,
      c = 1
    };

    MultiEnumSelection es({"a", "b", "c"}, {5, 2, 1}, {"a", "c"});
    auto value = es.as<SomeEnum>();
    REQUIRE(value.size() == 2);
    REQUIRE(value[0] == SomeEnum::a);
    REQUIRE(value[1] == SomeEnum::c);
  }

  SECTION("stringify")
  {
    enum class SomeEnum : int
    {
      a = 5,
      b = 2,
      c = 1
    };

    MultiEnumSelection es({"a", "b", "c"}, {5, 2, 1}, {"a", "b"});
    REQUIRE(utils::stringify(es) == "(a, b)");
  }

  SECTION("Errors")
  {
    REQUIRE_THROWS_WITH(
        MultiEnumSelection({"a", "b", "c"}, {"d"}),
        Catch::Matchers::ContainsSubstring("Invalid selection for MultiEnumSelection"));
    REQUIRE_THROWS_WITH(
        MultiEnumSelection({"a", "b", "c"}, {1, 2, 3}, {"d"}),
        Catch::Matchers::ContainsSubstring("Invalid selection for MultiEnumSelection"));
  }
}
