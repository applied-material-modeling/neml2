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
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/TensorValue.h"

using namespace neml2;

TEST_CASE("ParameterStore", "[models]")
{
  auto model = load_model("models/solid_mechanics/elasticity/LinearIsotropicElasticity.i", "model");
  auto batch_shape = TensorShape{5, 2};

  SECTION("class ParameterStore")
  {
    SECTION("named_parameters")
    {
      auto & params = model->named_parameters();

      REQUIRE(params.size() == 2);
      REQUIRE(params.count("E"));
      REQUIRE(params.count("nu"));
    }

    SECTION("get_parameter")
    {
      auto & params = model->named_parameters();
      auto & E = model->get_parameter("E");
      auto & nu = model->get_parameter("nu");

      REQUIRE(Tensor(E).batch_sizes() == TensorShape());
      REQUIRE(Tensor(nu).batch_sizes() == TensorShape());

      REQUIRE(E.type() == TensorType::kScalar);
      REQUIRE(nu.type() == TensorType::kScalar);

      // Modifying the individual parameter references should affect values stored in the parameter
      // dictionary.
      E = Scalar::full({1, 2}, {}, 1.0);
      nu = Scalar::full({5, 1}, {}, 0.3);
      REQUIRE(Tensor(*params["E"]).batch_sizes() == TensorShape{1, 2});
      REQUIRE(Tensor(*params["nu"]).batch_sizes() == TensorShape{5, 1});

      // Same thing say when the user wants to use torch AD
      E.requires_grad_(true);
      nu.requires_grad_(true);
      REQUIRE(Tensor(*params["E"]).requires_grad());
      REQUIRE(Tensor(*params["nu"]).requires_grad());
    }
  }
}

TEST_CASE("Nested parameter registration")
{
  auto model = load_model("models/test_ParameterStore.i", "model");

  const auto & params = model->named_parameters();
  REQUIRE(params.count("E1::value"));
  REQUIRE(params.count("E2::value"));
  REQUIRE(params.count("E3::value"));
  REQUIRE(params.count("elasticity1::nu"));
  REQUIRE(params.count("elasticity2::nu"));
  REQUIRE(params.count("elasticity2_another::nu"));
  REQUIRE(params.count("elasticity3::nu"));
}
