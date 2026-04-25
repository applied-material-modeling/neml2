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
#include "neml2/neml2.h"
#include "neml2/models/Variable.h"
#include "neml2/tensors/tensors.h"
#include "neml2/drivers/Driver.h"

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  SECTION("set_output_derivative_filter")
  {
    auto model = load_model("models/common/ScalarLinearCombination.i", "model");

    set_default_dtype(kFloat64);
    ValueMap in;
    in["A"] = Scalar::full(3.0);
    in["B"] = Scalar::full(2.0);

    SECTION("no filter returns all derivatives")
    {
      auto derivs = model->dvalue(in);
      REQUIRE(derivs.count("C") == 1);
      REQUIRE(derivs.at("C").count("A") == 1);
      REQUIRE(derivs.at("C").count("B") == 1);
    }

    SECTION("filter restricts to requested pair")
    {
      model->set_output_derivative_filter({{"C", "A"}});
      auto derivs = model->dvalue(in);
      REQUIRE(derivs.count("C") == 1);
      REQUIRE(derivs.at("C").count("A") == 1);
      REQUIRE(derivs.at("C").count("B") == 0);
    }

    SECTION("clearing filter restores all derivatives")
    {
      model->set_output_derivative_filter({{"C", "A"}});
      model->set_output_derivative_filter({});
      auto derivs = model->dvalue(in);
      REQUIRE(derivs.count("C") == 1);
      REQUIRE(derivs.at("C").count("A") == 1);
      REQUIRE(derivs.at("C").count("B") == 1);
    }

    SECTION("filter applies to value_and_dvalue")
    {
      model->set_output_derivative_filter({{"C", "B"}});
      auto [vals, derivs] = model->value_and_dvalue(in);
      REQUIRE(vals.count("C") == 1);
      REQUIRE(derivs.count("C") == 1);
      REQUIRE(derivs.at("C").count("A") == 0);
      REQUIRE(derivs.at("C").count("B") == 1);
    }

    SECTION("changing filter after evaluation uses the new filter")
    {
      // First evaluation with filter on A — traces the JIT graph
      model->set_output_derivative_filter({{"C", "A"}});
      {
        auto derivs = model->dvalue(in);
        REQUIRE(derivs.at("C").count("A") == 1);
        REQUIRE(derivs.at("C").count("B") == 0);
      }

      // Change filter to B — must invalidate the old traced graph and re-trace
      model->set_output_derivative_filter({{"C", "B"}});
      {
        auto derivs = model->dvalue(in);
        REQUIRE(derivs.at("C").count("A") == 0);
        REQUIRE(derivs.at("C").count("B") == 1);
      }
    }
  }

  SECTION("variable type")
  {
    auto model = load_model("models/common/ComposedModel3.i", "model");

    REQUIRE(model->input_variable("t").type() == TensorType::kScalar);
    REQUIRE(model->input_variable("T").type() == TensorType::kScalar);
    REQUIRE(model->input_variable("bar").type() == TensorType::kScalar);
    REQUIRE(model->input_variable("baz").type() == TensorType::kSR2);
    REQUIRE(model->input_variable("foo").type() == TensorType::kScalar);
    REQUIRE(model->output_variable("sum").type() == TensorType::kScalar);

    REQUIRE(utils::stringify(model->input_variable("t").type()) == "Scalar");
    REQUIRE(utils::stringify(model->input_variable("T").type()) == "Scalar");
    REQUIRE(utils::stringify(model->input_variable("bar").type()) == "Scalar");
    REQUIRE(utils::stringify(model->input_variable("baz").type()) == "SR2");
    REQUIRE(utils::stringify(model->input_variable("foo").type()) == "Scalar");
    REQUIRE(utils::stringify(model->output_variable("sum").type()) == "Scalar");
  }

  SECTION("graph execution exception")
  {
    auto factory = load_input("models/test_graph_execution_exception.i");
    auto driver = factory->get_driver("driver");
    REQUIRE_THROWS_WITH(
        driver->run(),
        Catch::Matchers::ContainsSubstring(
            "Try turning off just-in-time (JIT) compilation to get more detailed error messages."));
  }
}
