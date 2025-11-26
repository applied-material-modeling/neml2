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

#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/dev.h"
#include "neml2/tensors/functions/tr.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("dev", "[tensors/functions]", R2, SR2)
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();
  auto shape = test::generate_tensor_shape<TestType>();
  DYNAMIC_SECTION(cfg.desc() << " Shape: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto r1 = a - (1.0 / 3.0) * neml2::tr(a) * TestType::identity(a.options());
    auto r2 = neml2::dev(a);

    REQUIRE(test::match_tensor_shape(r2, shape));
    REQUIRE_THAT(r2, test::allclose(r1));
  }
}

TEST_CASE("dev - WR2", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();
  auto shape = test::generate_tensor_shape<WR2>();
  DYNAMIC_SECTION(cfg.desc() << " Shape: " << shape.desc())
  {
    auto a = test::generate_random_tensor<WR2>(cfg, shape);
    auto r1 = a;
    REQUIRE(test::match_tensor_shape(r1, shape));
    REQUIRE_THAT(neml2::dev(a), test::allclose(r1));
  }
}