// Copyright 2023, UChicago Argonne, LLC
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
#include <catch2/catch.hpp>

#include "neml2/base/ParameterSet.h"
#include "neml2/models/solid_mechanics/LinearIsotropicHardening.h"

using namespace neml2;

TEST_CASE("get and set", "[ParameterSet]")
{
  ParameterSet params;
  params.set<double>("p1") = 1.5;
  params.set<std::string>("p2") = "foo";
  params.set<unsigned int>("p3") = 3;
  params.set<std::vector<std::string>>("p4") = {"a", "b", "c", "d", "e"};
  params.set<std::vector<double>>("p5") = {1.2, -1.1, 100, 5.3};

  SECTION("size") { REQUIRE(params.size() == 5); }

  SECTION("get")
  {
    REQUIRE(params.get<double>("p1") == Approx(1.5));
    REQUIRE(params.get<std::string>("p2") == "foo");
    REQUIRE(params.get<unsigned int>("p3") == 3);
    REQUIRE_THAT(params.get<std::vector<std::string>>("p4"),
                 Catch::Matchers::Equals(std::vector<std::string>{"a", "b", "c", "d", "e"}));
    REQUIRE_THAT(params.get<std::vector<double>>("p5"),
                 Catch::Matchers::Approx(std::vector<double>{1.2, -1.1, 100, 5.3}));
  }
}
