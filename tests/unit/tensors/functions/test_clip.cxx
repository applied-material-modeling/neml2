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

#include "neml2/tensors/functions/clip.h"
#include "neml2/tensors/functions/clamp.h"
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/tensors.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("clip", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();
  auto shape = test::generate_tensor_shape<TestType>();

  DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
  {
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto lb = test::generate_random_tensor<TestType>(cfg, shape);
    auto ub = test::generate_random_tensor<TestType>(cfg, shape);

    ub = ub + neml2::abs(lb);

    auto c1 = neml2::clip(a, lb, ub);
    REQUIRE(test::match_tensor_shape(c1, shape));

    auto ref1 = neml2::clamp(a, lb, ub);
    REQUIRE_THAT(c1, test::allclose(ref1));

    const CScalar lower(-0.3);
    const CScalar upper(0.4);
    auto c2 = neml2::clip(a, lower, upper);
    REQUIRE(test::match_tensor_shape(c2, shape));

    auto ref2 = neml2::clamp(a, lower, upper);
    REQUIRE_THAT(c2, test::allclose(ref2));
  }
}
