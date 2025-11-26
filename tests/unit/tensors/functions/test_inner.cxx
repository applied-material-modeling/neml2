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
#include "neml2/tensors/functions/inner.h"
#include "neml2/tensors/functions/sum.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("inner", "[tensors/functions]", FOR_ALL_TENSORBASE_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);

  // Use only floating-point dtypes (linalg_vecdot does not support integers)
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::generate_tensor_shape<TestType>();

  DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
  {
    // Generate random tensors a and b
    auto a = test::generate_random_tensor<TestType>(cfg, shape);
    auto b = test::generate_random_tensor<TestType>(cfg, shape);

    // NEML2
    auto c_neml2 = neml2::inner(a, b);

    // Ref
    auto a_flat = a.base_flatten();
    auto b_flat = b.base_flatten();
    auto c_ref = base_sum(a_flat * b_flat);

    REQUIRE(c_neml2.base_dim() == 0);
    REQUIRE(c_ref.base_dim() == 0);
    // Dynamic shape should be the broadcast of a & b
    auto dyn_expected =
        neml2::utils::broadcast_dynamic_sizes(std::vector<Tensor>{Tensor(a), Tensor(b)});
    REQUIRE(c_neml2.dynamic_sizes() == dyn_expected);
    REQUIRE(c_ref.dynamic_sizes() == dyn_expected);

    REQUIRE_THAT(c_neml2, test::allclose(c_ref));
  }
}
