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

#include "neml2/dispatchers/derivmap_helpers.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"

using namespace neml2;

TEST_CASE("derivmap_helpers", "[dispatchers]")
{
  const auto scalar1_name = VariableName{"state", "s1"};
  const auto scalar2_name = VariableName{"state", "s2"};
  const auto deriv_value_1 = Scalar::full(300).dynamic_expand({5, 10});
  const auto deriv_value_2 = Scalar::full(500).dynamic_expand({7, 10});
  auto dvalue_map_1 = DerivMap({{scalar2_name, ValueMap({{scalar1_name, deriv_value_1}})}});
  auto dvalue_map_2 = DerivMap({{scalar2_name, ValueMap({{scalar1_name, deriv_value_2}})}});

  SECTION("cat")
  {
    auto result = derivmap_cat_reduce({dvalue_map_1, dvalue_map_2}, 0);
    REQUIRE(result[scalar2_name][scalar1_name].sizes() == TensorShape({12, 10}));
  }
  SECTION("no operation")
  {
    auto result = derivmap_no_operation(std::move(dvalue_map_1));
    REQUIRE(result.size() == 1);
    REQUIRE(result[scalar2_name].size() == 1);
    REQUIRE(result[scalar2_name][scalar1_name].sizes() == TensorShape({5, 10}));
  }
  /* Fill this in when we have GPU tests
  SECTION("move device")
  {

  }
  */
}
