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

#include "neml2/base/OptionSet.h"
#include "neml2/misc/string_utils.h"

using namespace neml2;

TEST_CASE("OptionSet", "[base]")
{
  SECTION("class OptionSet")
  {
    auto options = OptionSet();
    options.set<double>("p1") = 1.5;
    options.set<std::string>("p2") = "foo";
    options.set<unsigned int>("p3") = 3;
    options.set<std::vector<std::string>>("p4") = {"a", "b", "c", "d", "e"};
    options.set<std::vector<double>>("p5") = {1.2, -1.1, 100, 5.3};

    SECTION("size") { REQUIRE(options.size() == 5); }

    SECTION("get")
    {
      REQUIRE(options.get<double>("p1") == Catch::Approx(1.5));
      REQUIRE(options.get<std::string>("p2") == "foo");
      REQUIRE(options.get<unsigned int>("p3") == 3);
      REQUIRE_THAT(options.get<std::vector<std::string>>("p4"),
                   Catch::Matchers::Equals(std::vector<std::string>{"a", "b", "c", "d", "e"}));
      REQUIRE_THAT(options.get<std::vector<double>>("p5"),
                   Catch::Matchers::Approx(std::vector<double>{1.2, -1.1, 100, 5.3}));

      // The non-template version should allow me to retrieve metadata
      REQUIRE(options.get("p1").name() == "p1");
      REQUIRE(options.get("p1").type() == utils::demangle(typeid(double).name()));
      REQUIRE(options.get("p1").doc() == "");
      REQUIRE(!options.get("p1").suppressed());
    }

    SECTION("set")
    {
      // The non-template version should allome to alter the metadata
      std::string docstr = "This is an option named p1 with type double.";
      options.set("p1").doc() = docstr;
      options.set("p1").suppressed() = true;
      REQUIRE(options.get("p1").doc() == docstr);
      REQUIRE(options.get("p1").suppressed());
    }

    SECTION("copy")
    {
      OptionSet options2(options);
      REQUIRE(options2.get<double>("p1") == Catch::Approx(1.5));
      REQUIRE(options2.get<std::string>("p2") == "foo");
      REQUIRE(options2.get<unsigned int>("p3") == 3);
      REQUIRE_THAT(options2.get<std::vector<std::string>>("p4"),
                   Catch::Matchers::Equals(std::vector<std::string>{"a", "b", "c", "d", "e"}));
      REQUIRE_THAT(options2.get<std::vector<double>>("p5"),
                   Catch::Matchers::Approx(std::vector<double>{1.2, -1.1, 100, 5.3}));
    }

    SECTION("contains")
    {
      REQUIRE(options.contains("p1"));
      REQUIRE(options.contains("p2"));
      REQUIRE(!options.contains("p6"));
    }
  }
}
