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

#include "utils.h"
#include "neml2/models/Model.h"

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  SECTION("variable type")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");

    REQUIRE(model.input_variable({"forces", "t"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"forces", "temperature"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"old_forces", "t"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"old_state", "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"old_state", "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.input_variable({"old_state", "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"state", "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({"state", "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.input_variable({"state", "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.output_variable({"state", "sum"}).type() == TensorType::kScalar);

    REQUIRE(utils::stringify(model.input_variable({"forces", "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"forces", "temperature"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"old_forces", "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"old_state", "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"old_state", "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.input_variable({"old_state", "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"state", "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({"state", "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.input_variable({"state", "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.output_variable({"state", "sum"}).type()) == "Scalar");
  }

  SECTION("diagnose")
  {
    SECTION("input variables")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose1.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "Input variable whatever/foo_rate must be on one of the following sub-axes"));
    }
    SECTION("output variables")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose2.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "Output variable whatever/foo must be on one of the following sub-axes"));
    }
    SECTION("statefulness")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose1.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring("Input axis has old state variable foo, but the "
                                             "corresponding output state variable doesn't exist"));
    }
    SECTION("nonlinear system")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose3.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "This model is part of a nonlinear system. At least one of the input variables is "
              "solve-dependent, so all output variables MUST be solve-dependent"));
    }
  }
}
