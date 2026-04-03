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

#include "neml2/equation_systems/AxisLayout.h"

using namespace neml2;

TEST_CASE("AxisLayout", "[equation_systems]")
{
  // A 3-variable, 2-group layout:
  //   group 0: "a" (intmd={2}, base={3}), "b" (intmd={}, base={6})
  //   group 1: "c" (intmd={}, base={})
  const AxisLayout layout({{VariableName("a"), VariableName("b")}, {VariableName("c")}},
                          {TensorShape{2}, TensorShape{}, TensorShape{}},
                          {TensorShape{3}, TensorShape{6}, TensorShape{}},
                          {AxisLayout::IStructure::DENSE, AxisLayout::IStructure::BLOCK});

  SECTION("ngroup") { REQUIRE(layout.ngroup() == 2); }

  SECTION("group_offsets")
  {
    const auto [s0, e0] = layout.group_offsets(0);
    REQUIRE(s0 == 0);
    REQUIRE(e0 == 2);

    const auto [s1, e1] = layout.group_offsets(1);
    REQUIRE(s1 == 2);
    REQUIRE(e1 == 3);
  }

  SECTION("istr")
  {
    REQUIRE(layout.istr(0) == AxisLayout::IStructure::DENSE);
    REQUIRE(layout.istr(1) == AxisLayout::IStructure::BLOCK);
  }

  SECTION("group")
  {
    SECTION("group 0")
    {
      const auto g = layout.group(0);
      REQUIRE(g.nvar() == 2);
      REQUIRE(g.is_view());
      REQUIRE(g.var(0) == VariableName("a"));
      REQUIRE(g.var(1) == VariableName("b"));
      REQUIRE(g.intmd_sizes(0) == TensorShape{2});
      REQUIRE(g.base_sizes(0) == TensorShape{3});
      REQUIRE(g.intmd_sizes(1) == TensorShape{});
      REQUIRE(g.base_sizes(1) == TensorShape{6});
    }

    SECTION("group 1")
    {
      const auto g = layout.group(1);
      REQUIRE(g.nvar() == 1);
      REQUIRE(g.is_view());
      REQUIRE(g.var(0) == VariableName("c"));
      REQUIRE(g.intmd_sizes(0) == TensorShape{});
      REQUIRE(g.base_sizes(0) == TensorShape{});
    }

    // group() of a group view produces a further narrowed view with correct variables
    SECTION("group of group")
    {
      const auto g = layout.view().group(0);
      REQUIRE(g.nvar() == 2);
      REQUIRE(g.var(0) == VariableName("a"));
    }
  }

  SECTION("view")
  {
    const auto v = layout.view();
    REQUIRE(v.is_view());
    // view() preserves group structure, so ngroup() still works
    REQUIRE(v.ngroup() == 2);
    REQUIRE(v.nvar() == 3);
    REQUIRE(v.var(0) == VariableName("a"));
    REQUIRE(v.var(1) == VariableName("b"));
    REQUIRE(v.var(2) == VariableName("c"));
  }

  SECTION("is_view")
  {
    REQUIRE(!layout.is_view());
    REQUIRE(layout.view().is_view());
    REQUIRE(layout.group(0).is_view());
  }

  SECTION("nvar")
  {
    REQUIRE(layout.nvar() == 3);
    REQUIRE(layout.group(0).nvar() == 2);
    REQUIRE(layout.group(1).nvar() == 1);
    REQUIRE(layout.view().nvar() == 3);
  }

  SECTION("storage_sizes")
  {
    SECTION("exclude intermediate")
    {
      // numel of base_sizes only: {3}->3, {6}->6, {}->1
      const auto ss = layout.storage_sizes(/*include_intmd=*/false);
      REQUIRE(ss.size() == 3);
      REQUIRE(ss[0] == 3);
      REQUIRE(ss[1] == 6);
      REQUIRE(ss[2] == 1);
    }

    SECTION("include intermediate")
    {
      // numel(intmd) * numel(base): 2*3=6, 1*6=6, 1*1=1
      const auto ss = layout.storage_sizes(/*include_intmd=*/true);
      REQUIRE(ss.size() == 3);
      REQUIRE(ss[0] == 6);
      REQUIRE(ss[1] == 6);
      REQUIRE(ss[2] == 1);
    }

    SECTION("group view")
    {
      // storage_sizes on a sub-group view only covers that group's variables
      const auto ss = layout.group(0).storage_sizes(/*include_intmd=*/false);
      REQUIRE(ss.size() == 2);
      REQUIRE(ss[0] == 3);
      REQUIRE(ss[1] == 6);
    }
  }

  SECTION("var")
  {
    REQUIRE(layout.var(0) == VariableName("a"));
    REQUIRE(layout.var(1) == VariableName("b"));
    REQUIRE(layout.var(2) == VariableName("c"));

    // var() works through views
    REQUIRE(layout.group(0).var(0) == VariableName("a"));
    REQUIRE(layout.group(1).var(0) == VariableName("c"));
  }

  SECTION("intmd_sizes")
  {
    REQUIRE(layout.intmd_sizes(0) == TensorShape{2});
    REQUIRE(layout.intmd_sizes(1) == TensorShape{});
    REQUIRE(layout.intmd_sizes(2) == TensorShape{});

    // intmd_sizes() works through views
    REQUIRE(layout.group(0).intmd_sizes(0) == TensorShape{2});
    REQUIRE(layout.group(0).intmd_sizes(1) == TensorShape{});
    REQUIRE(layout.group(1).intmd_sizes(0) == TensorShape{});
  }

  SECTION("base_sizes")
  {
    REQUIRE(layout.base_sizes(0) == TensorShape{3});
    REQUIRE(layout.base_sizes(1) == TensorShape{6});
    REQUIRE(layout.base_sizes(2) == TensorShape{});

    // base_sizes() works through views
    REQUIRE(layout.group(0).base_sizes(0) == TensorShape{3});
    REQUIRE(layout.group(0).base_sizes(1) == TensorShape{6});
    REQUIRE(layout.group(1).base_sizes(0) == TensorShape{});
  }
}
