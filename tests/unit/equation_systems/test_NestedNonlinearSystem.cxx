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

TEST_CASE("NestedNonlinearSystem", "[equation_systems]")
{
  SECTION("two groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("two_groups");
    REQUIRE(eq_sys != nullptr);

    REQUIRE(eq_sys->n_ugroup() == 2);

    const auto & umap0 = eq_sys->umap(0);
    REQUIRE(umap0.size() == 2);
    REQUIRE(umap0[0] == VariableName("state", "foo"));
    REQUIRE(umap0[1] == VariableName("state", "bar"));

    const auto & ulayout0 = eq_sys->ulayout(0);
    REQUIRE(ulayout0.size() == 2);
    REQUIRE(ulayout0[0] == TensorShape{});
    REQUIRE(ulayout0[1] == TensorShape{});

    const auto & bmap0 = eq_sys->bmap(0);
    REQUIRE(bmap0.size() == 2);
    REQUIRE(bmap0[0] == VariableName("residual", "foo"));
    REQUIRE(bmap0[1] == VariableName("residual", "bar"));

    const auto & blayout0 = eq_sys->blayout(0);
    REQUIRE(blayout0.size() == 2);
    REQUIRE(blayout0[0] == TensorShape{});
    REQUIRE(blayout0[1] == TensorShape{});

    const auto & umap1 = eq_sys->umap(1);
    REQUIRE(umap1.size() == 1);
    REQUIRE(umap1[0] == VariableName("state", "baz"));

    const auto & ulayout1 = eq_sys->ulayout(1);
    REQUIRE(ulayout1.size() == 1);
    REQUIRE(ulayout1[0] == TensorShape{6});

    const auto & bmap1 = eq_sys->bmap(1);
    REQUIRE(bmap1.size() == 1);
    REQUIRE(bmap1[0] == VariableName("residual", "baz"));

    const auto & blayout1 = eq_sys->blayout(1);
    REQUIRE(blayout1.size() == 1);
    REQUIRE(blayout1[0] == TensorShape{6});

    // Intermediate layouts are lazily initialized after first assembly.
  }

  SECTION("three groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("three_groups");
    REQUIRE(eq_sys != nullptr);

    REQUIRE(eq_sys->n_ugroup() == 3);

    const auto & umap0 = eq_sys->umap(0);
    REQUIRE(umap0.size() == 1);
    REQUIRE(umap0[0] == VariableName("state", "foo"));

    const auto & umap1 = eq_sys->umap(1);
    REQUIRE(umap1.size() == 1);
    REQUIRE(umap1[0] == VariableName("state", "bar"));

    const auto & umap2 = eq_sys->umap(2);
    REQUIRE(umap2.size() == 1);
    REQUIRE(umap2[0] == VariableName("state", "baz"));
  }

  SECTION("single group")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("single_group");
    REQUIRE(eq_sys != nullptr);

    REQUIRE(eq_sys->n_ugroup() == 1);

    const auto & umap0 = eq_sys->umap(0);
    REQUIRE(umap0.size() == 3);
    REQUIRE(umap0[0] == VariableName("state", "foo"));
    REQUIRE(umap0[1] == VariableName("state", "bar"));
    REQUIRE(umap0[2] == VariableName("state", "baz"));
  }

  SECTION("reordered groups")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("reordered");
    REQUIRE(eq_sys != nullptr);

    REQUIRE(eq_sys->n_ugroup() == 2);

    const auto & umap0 = eq_sys->umap(0);
    REQUIRE(umap0.size() == 1);
    REQUIRE(umap0[0] == VariableName("state", "baz"));

    const auto & umap1 = eq_sys->umap(1);
    REQUIRE(umap1.size() == 2);
    REQUIRE(umap1[0] == VariableName("state", "bar"));
    REQUIRE(umap1[1] == VariableName("state", "foo"));

    const auto & bmap0 = eq_sys->bmap(0);
    REQUIRE(bmap0.size() == 1);
    REQUIRE(bmap0[0] == VariableName("residual", "baz"));

    const auto & bmap1 = eq_sys->bmap(1);
    REQUIRE(bmap1.size() == 2);
    REQUIRE(bmap1[0] == VariableName("residual", "bar"));
    REQUIRE(bmap1[1] == VariableName("residual", "foo"));
  }

  SECTION("invalid group_id throws")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem.i");
    auto eq_sys = factory->get_es<ModelNonlinearSystem>("two_groups");
    REQUIRE(eq_sys != nullptr);

    REQUIRE_THROWS_AS(eq_sys->umap(2), NEMLException);
    REQUIRE_THROWS_AS(eq_sys->ulayout(2), NEMLException);
    REQUIRE_THROWS_AS(eq_sys->bmap(2), NEMLException);
    REQUIRE_THROWS_AS(eq_sys->blayout(2), NEMLException);

    REQUIRE_THROWS_AS(eq_sys->umap(100), NEMLException);
  }

  SECTION("missing state variable throws")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem_errors.i");
    REQUIRE_THROWS_WITH(factory->get_es<ModelNonlinearSystem>("missing_variable"),
                        Catch::Matchers::ContainsSubstring("is not included in 'unknown_groups'"));
  }

  SECTION("nonexistent variable throws")
  {
    auto factory = load_input("equation_systems/test_NestedNonlinearSystem_errors.i");
    REQUIRE_THROWS_WITH(factory->get_es<ModelNonlinearSystem>("nonexistent_variable"),
                        Catch::Matchers::ContainsSubstring("is not a state variable"));
  }
}
