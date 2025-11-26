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
#include <catch2/catch_template_test_macros.hpp>

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/cat.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("dynamic_cat", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  // Need some dyanmic size to test cat
  std::vector<neml2::TensorShape> dynamic_shapes = {neml2::TensorShape{2, 3},
                                                    neml2::TensorShape{4, 3}};
  auto shape = test::generate_tensor_shape<TestType>(dynamic_shapes);
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc() << " s2: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto b = test::generate_random_tensor<TestType>(cfg, shape);
    auto c = neml2::dynamic_cat({a, b}, /*dim=*/0);

    auto a1 = a.dynamic_sizes().concrete()[0];
    auto b1 = b.dynamic_sizes().concrete()[0];
    shape.dynamic_sizes[0] += shape.dynamic_sizes[0];
    REQUIRE(test::match_tensor_shape(c, shape));

    REQUIRE_THAT(c.dynamic_index({indexing::Slice(0, a1)}), test::allclose(a));
    REQUIRE_THAT(c.dynamic_index({indexing::Slice(a1, a1 + b1)}), test::allclose(b));
  }
}

TEMPLATE_TEST_CASE("intmd_cat", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  // Need some intmd size to test cat
  std::vector<neml2::TensorShape> intmd_shapes = {neml2::TensorShape{2, 3},
                                                  neml2::TensorShape{4, 3}};
  auto shape = test::generate_tensor_shape<TestType>(std::nullopt, intmd_shapes);
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc() << " s2: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto b = test::generate_random_tensor<TestType>(cfg, shape);
    auto c = neml2::intmd_cat({a, b}, /*dim=*/0);

    auto a1 = a.intmd_sizes()[0];
    auto b1 = b.intmd_sizes()[0];
    shape.intmd_sizes[0] += shape.intmd_sizes[0];
    REQUIRE(test::match_tensor_shape(c, shape));

    REQUIRE_THAT(c.intmd_index({indexing::Slice(0, a1)}), test::allclose(a));
    REQUIRE_THAT(c.intmd_index({indexing::Slice(a1, a1 + b1)}), test::allclose(b));
  }
}

TEST_CASE("base_cat", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<Vec>();
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc() << " s2: " << shape.desc())
  {
    auto a = test::generate_random_tensor<Vec>(cfg, shape);
    auto b = test::generate_random_tensor<Vec>(cfg, shape);
    auto c = neml2::base_cat({a, b}, /*dim=*/0);

    auto a1 = a.base_sizes()[0];
    auto b1 = b.base_sizes()[0];
    shape.base_sizes[0] += shape.base_sizes[0];
    REQUIRE(test::match_tensor_shape(c, shape));

    REQUIRE_THAT(c.base_index({indexing::Slice(0, a1)}), test::allclose(a));
    REQUIRE_THAT(c.base_index({indexing::Slice(a1, a1 + b1)}), test::allclose(b));
  }
}