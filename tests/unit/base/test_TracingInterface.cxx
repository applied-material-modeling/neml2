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
#include <memory>

#include "SampleEventTracer.h"

using namespace neml2;

#ifdef NEML2_HAS_JSON
TEST_CASE("TracingInterface", "[base]")
{
  SECTION("single writer")
  {
    event_trace_writers().clear();
    auto tracer = std::make_unique<SampleEventTracer>(
        "base/test_tracer.json", "duration_event", "instant_event");
    tracer->dump();
    tracer.reset();
    event_trace_writers().clear();

    // "test_tracer.json" should have 5 events
    // 0: B, trace writer
    // 1: B, duration_event
    // 2: i, instant_event
    // 3: E, duration_event
    // 4: E, trace writer
    std::ifstream f("base/test_tracer.json");
    auto data = json::parse(f);
    f.close();
    REQUIRE(data.size() == 5);
    REQUIRE(data[0]["name"] == "trace writer");
    REQUIRE(data[0]["ph"] == "B");
    REQUIRE(data[1]["name"] == "duration_event");
    REQUIRE(data[1]["ph"] == "B");
    REQUIRE(data[2]["name"] == "instant_event");
    REQUIRE(data[2]["ph"] == "i");
    REQUIRE(data[3]["name"] == "duration_event");
    REQUIRE(data[3]["ph"] == "E");
    REQUIRE(data[4]["name"] == "trace writer");
    REQUIRE(data[4]["ph"] == "E");
  }

  SECTION("multiple writers")
  {
    event_trace_writers().clear();
    std::vector<std::unique_ptr<SampleEventTracer>> tracers;
    tracers.emplace_back(
        std::make_unique<SampleEventTracer>("base/test_tracer_1.json", "d1", "i1"));
    tracers.emplace_back(
        std::make_unique<SampleEventTracer>("base/test_tracer_2.json", "d2", "i2"));
    tracers.emplace_back(
        std::make_unique<SampleEventTracer>("base/test_tracer_1.json", "d3", "i3"));
    tracers[0]->dump();
    tracers[1]->dump();
    tracers[2]->dump();
    tracers.clear();
    event_trace_writers().clear();

    // tracers[0] and tracers[2] should write to the same file "test_tracer_1.json"
    // tracers[1] should write to a different file

    // "test_tracer_1.json" should have 8 events
    // 0: B, trace writer
    // 1: B, d1
    // 2: i, i1
    // 3: E, d1
    // 4: B, d3
    // 5: i, i3
    // 6: E, d3
    // 7: E, trace writer
    std::ifstream f1("base/test_tracer_1.json");
    auto data1 = json::parse(f1);
    f1.close();
    REQUIRE(data1.size() == 8);
    REQUIRE(data1[0]["name"] == "trace writer");
    REQUIRE(data1[0]["ph"] == "B");
    REQUIRE(data1[1]["name"] == "d1");
    REQUIRE(data1[1]["ph"] == "B");
    REQUIRE(data1[2]["name"] == "i1");
    REQUIRE(data1[2]["ph"] == "i");
    REQUIRE(data1[3]["name"] == "d1");
    REQUIRE(data1[3]["ph"] == "E");
    REQUIRE(data1[4]["name"] == "d3");
    REQUIRE(data1[4]["ph"] == "B");
    REQUIRE(data1[5]["name"] == "i3");
    REQUIRE(data1[5]["ph"] == "i");
    REQUIRE(data1[6]["name"] == "d3");
    REQUIRE(data1[6]["ph"] == "E");
    REQUIRE(data1[7]["name"] == "trace writer");
    REQUIRE(data1[7]["ph"] == "E");

    // "test_tracer_2.json" should have 5 events
    // 0: B, trace writer
    // 1: B, d2
    // 2: i, i2
    // 3: E, d2
    // 4: E, trace writer
    std::ifstream f2("base/test_tracer_2.json");
    auto data2 = json::parse(f2);
    f2.close();
    REQUIRE(data2.size() == 5);
    REQUIRE(data2[0]["name"] == "trace writer");
    REQUIRE(data2[0]["ph"] == "B");
    REQUIRE(data2[1]["name"] == "d2");
    REQUIRE(data2[1]["ph"] == "B");
    REQUIRE(data2[2]["name"] == "i2");
    REQUIRE(data2[2]["ph"] == "i");
    REQUIRE(data2[3]["name"] == "d2");
    REQUIRE(data2[3]["ph"] == "E");
    REQUIRE(data2[4]["name"] == "trace writer");
    REQUIRE(data2[4]["ph"] == "E");
  }
}
#endif
