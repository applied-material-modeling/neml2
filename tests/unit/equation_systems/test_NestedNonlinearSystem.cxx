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

#include "neml2/neml2.h"
#include "neml2/models/ModelNonlinearSystem.h"

using namespace neml2;

TEST_CASE("ModelNonlinearSystem", "[equation_systems]")
{
  SECTION("two groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("two_groups");
    REQUIRE(eq_sys != nullptr);

    // unknown layout: group 0 = {state/foo, state/bar}, group 1 = {state/baz}
    const auto & ul = *eq_sys->ulayout();
    REQUIRE(ul.ngroup() == 2);

    const auto ug0 = ul.group(0);
    REQUIRE(ug0.size() == 2);
    REQUIRE(ug0.var(0) == VariableName("state", "foo"));
    REQUIRE(ug0.var(1) == VariableName("state", "bar"));
    REQUIRE(ug0.base_sizes(0) == TensorShape{}); // Scalar
    REQUIRE(ug0.base_sizes(1) == TensorShape{}); // Scalar

    const auto ug1 = ul.group(1);
    REQUIRE(ug1.size() == 1);
    REQUIRE(ug1.var(0) == VariableName("state", "baz"));
    REQUIRE(ug1.base_sizes(0) == TensorShape{6}); // SR2

    // residual layout: group 0 = {residual/foo, residual/bar}, group 1 = {residual/baz}
    const auto & bl = *eq_sys->blayout();
    REQUIRE(bl.ngroup() == 2);

    const auto bg0 = bl.group(0);
    REQUIRE(bg0.size() == 2);
    REQUIRE(bg0.var(0) == VariableName("residual", "foo"));
    REQUIRE(bg0.var(1) == VariableName("residual", "bar"));
    REQUIRE(bg0.base_sizes(0) == TensorShape{}); // Scalar
    REQUIRE(bg0.base_sizes(1) == TensorShape{}); // Scalar

    const auto bg1 = bl.group(1);
    REQUIRE(bg1.size() == 1);
    REQUIRE(bg1.var(0) == VariableName("residual", "baz"));
    REQUIRE(bg1.base_sizes(0) == TensorShape{6}); // SR2
  }

  SECTION("three groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("three_groups");
    REQUIRE(eq_sys != nullptr);

    const auto & ul = *eq_sys->ulayout();
    REQUIRE(ul.ngroup() == 3);

    REQUIRE(ul.group(0).size() == 1);
    REQUIRE(ul.group(0).var(0) == VariableName("state", "foo"));

    REQUIRE(ul.group(1).size() == 1);
    REQUIRE(ul.group(1).var(0) == VariableName("state", "bar"));

    REQUIRE(ul.group(2).size() == 1);
    REQUIRE(ul.group(2).var(0) == VariableName("state", "baz"));
    REQUIRE(ul.group(2).base_sizes(0) == TensorShape{6}); // SR2
  }

  SECTION("single group")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("single_group");
    REQUIRE(eq_sys != nullptr);

    const auto & ul = *eq_sys->ulayout();
    REQUIRE(ul.ngroup() == 1);

    const auto g = ul.group(0);
    REQUIRE(g.size() == 3);
    REQUIRE(g.var(0) == VariableName("state", "foo"));
    REQUIRE(g.var(1) == VariableName("state", "bar"));
    REQUIRE(g.var(2) == VariableName("state", "baz"));
  }

  SECTION("reordered groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("reordered");
    REQUIRE(eq_sys != nullptr);

    const auto & ul = *eq_sys->ulayout();
    REQUIRE(ul.ngroup() == 2);

    const auto ug0 = ul.group(0);
    REQUIRE(ug0.size() == 1);
    REQUIRE(ug0.var(0) == VariableName("state", "baz"));

    const auto ug1 = ul.group(1);
    REQUIRE(ug1.size() == 2);
    REQUIRE(ug1.var(0) == VariableName("state", "bar"));
    REQUIRE(ug1.var(1) == VariableName("state", "foo"));

    const auto & bl = *eq_sys->blayout();

    const auto bg0 = bl.group(0);
    REQUIRE(bg0.size() == 1);
    REQUIRE(bg0.var(0) == VariableName("residual", "baz"));

    const auto bg1 = bl.group(1);
    REQUIRE(bg1.size() == 2);
    REQUIRE(bg1.var(0) == VariableName("residual", "bar"));
    REQUIRE(bg1.var(1) == VariableName("residual", "foo"));
  }

  SECTION("invalid group index throws")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("two_groups");
    REQUIRE(eq_sys != nullptr);

    REQUIRE_THROWS_AS(eq_sys->ulayout()->group(2), NEMLException);
    REQUIRE_THROWS_AS(eq_sys->blayout()->group(2), NEMLException);
    REQUIRE_THROWS_AS(eq_sys->ulayout()->group(100), NEMLException);
  }

  SECTION("nonexistent variable throws")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem_errors.i");
    REQUIRE_THROWS_WITH(factory->get_es<ModelNonlinearSystem>("nonexistent_variable"),
                        Catch::Matchers::ContainsSubstring("does not exist"));
  }
}
