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

#include "neml2/base/guards.h"

using namespace neml2;

TEST_CASE("guards", "[base]")
{
  SECTION("timed_sections")
  {
    // Clear any pre-existing entries to get a predictable state
    timed_sections().clear();
    REQUIRE(timed_sections().empty());
  }

  SECTION("TimedSection records an entry after destruction")
  {
    timed_sections().clear();

    // Entry must NOT be present before construction or during lifetime
    REQUIRE(timed_sections().count("forward") == 0);

    {
      TimedSection ts("my_model", "forward");
      // Not yet written — only written on destruction
      REQUIRE(timed_sections().count("forward") == 0);
    }

    // After the destructor runs the entry is present.
    // Note: timed_sections() maps [section][name].
    REQUIRE(timed_sections().count("forward") == 1);
    REQUIRE(timed_sections().at("forward").count("my_model") == 1);
  }

  SECTION("TimedSection accumulates time across multiple instances")
  {
    timed_sections().clear();

    {
      TimedSection ts("acc_test", "sec_a");
    }
    const auto first = timed_sections().at("sec_a").at("acc_test");

    {
      TimedSection ts("acc_test", "sec_a");
    }
    const auto second = timed_sections().at("sec_a").at("acc_test");

    // Time only accumulates (non-decreasing)
    REQUIRE(second >= first);
  }

  SECTION("TimedSection with distinct section names creates separate entries")
  {
    timed_sections().clear();

    {
      TimedSection ts("multi_model", "section_x");
    }
    {
      TimedSection ts("multi_model", "section_y");
    }

    REQUIRE(timed_sections().at("section_x").count("multi_model") == 1);
    REQUIRE(timed_sections().at("section_y").count("multi_model") == 1);
  }

  SECTION("Multiple model names are tracked independently")
  {
    timed_sections().clear();

    {
      TimedSection ts("model_a", "step");
    }
    {
      TimedSection ts("model_b", "step");
    }

    REQUIRE(timed_sections().count("step") == 1);
    REQUIRE(timed_sections().at("step").count("model_a") == 1);
    REQUIRE(timed_sections().at("step").count("model_b") == 1);
  }
}
