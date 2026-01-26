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
#include "neml2/tensors/functions/min.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("dynamic_min", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<TestType>();
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);

    if (a.dynamic_dim() == 0)
      return;

    auto b1 = neml2::dynamic_min(a, /*dim=*/0);
    auto b2 = std::get<0>(at::min(a, 0));

    REQUIRE_THAT(b1, test::allclose(b2));
  }
}

TEMPLATE_TEST_CASE("intmd_min", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<TestType>();
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);

    if (a.intmd_dim() == 0)
      return;

    auto b1 = neml2::intmd_min(a, /*dim=*/0);
    auto b2 = std::get<0>(at::min(a, a.dynamic_dim()));

    REQUIRE_THAT(b1, test::allclose(b2));
  }
}

TEST_CASE("base_min", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<Scalar>();
  DYNAMIC_SECTION(cfg.desc() << " s1: " << shape.desc())
  {
    auto a = test::generate_random_tensor<Scalar>(cfg, shape);
    if (a.base_dim() == 0)
      return;

    auto b1 = neml2::base_min(a, /*dim=*/0);
    auto b2 = std::get<0>(at::min(a, a.intmd_dim()));
    REQUIRE_THAT(b1, test::allclose(b2));
  }
}