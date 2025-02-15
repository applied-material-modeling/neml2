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

#include "neml2/models/LinearCombination.h"

using namespace neml2;
using ScalarLinearCombination = LinearCombination<Scalar>;

TEST_CASE("Factory", "[base]")
{
  auto options = ScalarLinearCombination::expected_options();
  options.name() = "example";
  options.type() = "ScalarLinearCombination";
  options.set<std::vector<VariableName>>("from_var") = {VariableName(STATE, "A"),
                                                        VariableName(STATE, "substate", "B")};
  options.set<VariableName>("to_var") = VariableName(STATE, "outsub", "C");

  OptionCollection all_options;
  all_options["Models"]["example"] = options;

  Factory::clear();
  Factory::load_options(all_options);

  SECTION("get_object")
  {
    auto & summodel = Factory::get_object<ScalarLinearCombination>("Models", "example");

    REQUIRE(summodel.input_axis().has_subaxis(STATE));
    REQUIRE(summodel.input_axis().subaxis(STATE).has_subaxis("substate"));
    REQUIRE(summodel.input_axis().subaxis(STATE).has_variable("A"));
    REQUIRE(summodel.input_axis().subaxis(STATE).subaxis("substate").has_variable("B"));

    REQUIRE(summodel.output_axis().has_subaxis(STATE));
    REQUIRE(summodel.output_axis().subaxis(STATE).has_subaxis("outsub"));
    REQUIRE(summodel.output_axis().subaxis(STATE).subaxis("outsub").has_variable("C"));
  }

  SECTION("loaded_options")
  {
    const auto & all_opts = Factory::loaded_options();
    const auto & opts = all_opts["Models"].at("example");
    REQUIRE(opts.contains("from_var"));
    REQUIRE(opts.contains("to_var"));
    REQUIRE(!opts.contains("non_existent"));
  }
}
