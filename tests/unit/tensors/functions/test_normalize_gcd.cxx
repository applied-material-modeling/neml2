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

#include "neml2/tensors/functions/normalize_gcd.h"
#include "neml2/tensors/tensors.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("normalize_gcd", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);

  // Test both integer and floating-point inputs
  auto cfg  = test::generate_tensor_config(test::fp_dtypes());
  auto shape   = test::generate_tensor_shape<TestType>();

  DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto b = neml2::normalize_gcd(a);

    REQUIRE(test::match_tensor_shape(b, shape));

    auto bf = b.base_flatten();
    const Size n = bf.base_size(0);

    // Distinguish dtype: keep Int as is, convert Float to Int
    auto convert = [](const at::Tensor &x) {
      return c10::isIntegralType(x.scalar_type(), false)
                ? x
                : x.to(get_default_integer_dtype());
    };

    // Take the first element as initial gcd
    at::Tensor d = convert(bf.base_index({0}));

    // Compute gcd across all base elements
    for (Size i = 1; i < n; ++i)
      d = at::gcd(d, convert(bf.base_index({i})));

    // Check if normalized: gcd should be ≈ 1 (exact 1 for integers)
    auto abs_d = at::abs(d.to(get_default_integer_dtype()));

    if (c10::isIntegralType(b.scalar_type(), false))
    {
      // Integer: gcd must equal 1 exactly
      REQUIRE(abs_d.min().item<long long>() == 1);
      REQUIRE(abs_d.max().item<long long>() == 1);
    }
    else
    {
      // Float: allow tiny deviation from 1
      auto max_d = abs_d.max().item<double>();
      auto min_d = abs_d.min().item<double>();
      REQUIRE(max_d / std::max(min_d, 1e-9) < 1.01);
    }
  }
}
