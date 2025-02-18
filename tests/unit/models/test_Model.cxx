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
#include "neml2/models/Variable.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  SECTION("variable type")
  {
    auto & model = reload_model("models/ComposedModel3.i", "model");

    REQUIRE(model.input_variable({FORCES, "t"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({FORCES, "temperature"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({OLD_FORCES, "t"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({OLD_STATE, "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({OLD_STATE, "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.input_variable({OLD_STATE, "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({STATE, "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.input_variable({STATE, "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.input_variable({STATE, "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.output_variable({STATE, "sum"}).type() == TensorType::kScalar);

    REQUIRE(utils::stringify(model.input_variable({FORCES, "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({FORCES, "temperature"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({OLD_FORCES, "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({OLD_STATE, "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({OLD_STATE, "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.input_variable({OLD_STATE, "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({STATE, "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.input_variable({STATE, "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.input_variable({STATE, "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.output_variable({STATE, "sum"}).type()) == "Scalar");
  }

  SECTION("diagnose")
  {
    SECTION("input variables")
    {
      auto & model = reload_model("models/test_Model_diagnose1.i", "model");
      auto diagnoses = diagnose(model);

      REQUIRE(diagnoses.size() == 2);
      REQUIRE_THAT(
          diagnoses[0].what(),
          Catch::Matchers::ContainsSubstring(
              "Input variable whatever/foo_rate must be on one of the following sub-axes"));
      REQUIRE_THAT(diagnoses[1].what(),
                   Catch::Matchers::ContainsSubstring(
                       "Variable whatever/foo_rate must be on the state sub-axis"));
    }

    SECTION("output variables")
    {
      auto & model = reload_model("models/test_Model_diagnose2.i", "model");
      auto diagnoses = diagnose(model);

      REQUIRE(diagnoses.size() == 1);
      REQUIRE_THAT(diagnoses[0].what(),
                   Catch::Matchers::ContainsSubstring(
                       "Output variable whatever/foo must be on one of the following sub-axes"));
    }

    SECTION("nonlinear system")
    {
      auto & model = reload_model("models/test_Model_diagnose3.i", "model");
      auto diagnoses = diagnose(model);

      REQUIRE(diagnoses.size() == 1);
      REQUIRE_THAT(
          diagnoses[0].what(),
          Catch::Matchers::ContainsSubstring(
              "This model is part of a nonlinear system. At least one of the input variables is "
              "solve-dependent, so all output variables MUST be solve-dependent"));
    }
  }
}
