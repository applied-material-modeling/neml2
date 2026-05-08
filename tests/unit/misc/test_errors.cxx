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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "neml2/misc/errors.h"

using namespace neml2;

TEST_CASE("errors", "[misc]")
{
  SECTION("NEMLException")
  {
    SECTION("default constructor")
    {
      NEMLException e;
      REQUIRE(std::string(e.what()) == "");
    }

    SECTION("message constructor")
    {
      NEMLException e("something went wrong");
      REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("something went wrong"));
    }

    SECTION("is std::exception")
    {
      NEMLException e("test");
      const std::exception & base = e;
      REQUIRE_THAT(base.what(), Catch::Matchers::ContainsSubstring("test"));
    }
  }

  SECTION("SetupException")
  {
    SECTION("message constructor")
    {
      SetupException e("setup failed");
      REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("setup failed"));
    }

    SECTION("is std::exception")
    {
      SetupException e("test");
      const std::exception & base = e;
      REQUIRE_THAT(base.what(), Catch::Matchers::ContainsSubstring("test"));
    }
  }

  SECTION("ParserException")
  {
    SECTION("message constructor")
    {
      ParserException e("parse error");
      REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("parse error"));
    }

    SECTION("is SetupException")
    {
      ParserException e("test");
      const SetupException & base = e;
      REQUIRE_THAT(base.what(), Catch::Matchers::ContainsSubstring("test"));
    }
  }

  SECTION("FactoryException")
  {
    SECTION("message constructor")
    {
      FactoryException e("factory error");
      REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("factory error"));
    }

    SECTION("is SetupException")
    {
      FactoryException e("test");
      const SetupException & base = e;
      REQUIRE_THAT(base.what(), Catch::Matchers::ContainsSubstring("test"));
    }
  }

  SECTION("Diagnosis")
  {
    SECTION("message constructor")
    {
      Diagnosis e("diagnosis message");
      REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("diagnosis message"));
    }

    SECTION("is NEMLException")
    {
      Diagnosis e("test");
      const NEMLException & base = e;
      REQUIRE_THAT(base.what(), Catch::Matchers::ContainsSubstring("test"));
    }
  }

  SECTION("throw and catch")
  {
    SECTION("catch NEMLException as std::exception")
    {
      REQUIRE_THROWS_AS([]() { throw NEMLException("neml error"); }(), NEMLException);
    }

    SECTION("catch ParserException as SetupException")
    {
      REQUIRE_THROWS_AS([]() { throw ParserException("parse"); }(), SetupException);
    }

    SECTION("catch FactoryException as SetupException")
    {
      REQUIRE_THROWS_AS([]() { throw FactoryException("factory"); }(), SetupException);
    }

    SECTION("catch Diagnosis as NEMLException")
    {
      REQUIRE_THROWS_AS([]() { throw Diagnosis("diag"); }(), NEMLException);
    }
  }
}
