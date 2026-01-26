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
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/base/Settings.h"
#include "neml2/base/HITParser.h"
#include "neml2/base/InputFile.h"
#include "neml2/neml2.h"

using namespace neml2;

TEST_CASE("Settings", "[Settings]")
{
  SECTION("parsing")
  {
    // Parse input file
    HITParser parser;
    auto input = parser.parse("base/test_HITParser1.i");

    // After applying the global settings
    const auto settings = *input.settings();
    REQUIRE(settings.buffer_name_separator() == "::");
    REQUIRE(settings.parameter_name_separator() == "::");
    REQUIRE(!settings.require_double_precision());
  }

  SECTION("disable_jit")
  {
    auto model1 = load_model("base/test_disable_jit.i", "model");
    REQUIRE(!model1->is_jit_enabled());

    auto model2 = load_model("base/test_disable_jit_error.i", "model");
    REQUIRE_THROWS_WITH(diagnose_and_throw(*model2),
                        Catch::Matchers::ContainsSubstring("JIT compilation is disabled globally"));
  }
}
