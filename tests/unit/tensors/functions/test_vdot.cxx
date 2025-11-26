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

#include "neml2/tensors/Vec.h"
#include "neml2/tensors/functions/vdot.h"
#include "neml2/tensors/functions/sum.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("vdot", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape1 = test::generate_tensor_shape<Vec>();
  auto shape2 = test::generate_tensor_shape<Vec>();

  DYNAMIC_SECTION(cfg.desc() << " LHS: " << shape1.desc() << " RHS: " << shape2.desc())
  {
    auto a = test::generate_random_tensor<Vec>(cfg, shape1);
    auto b = test::generate_random_tensor<Vec>(cfg, shape2);

    // NEML2 vdot
    auto c_neml2 = neml2::vdot(a, b);

    // reference implementation
    auto a_flat = a.base_flatten();
    auto b_flat = b.base_flatten();
    auto c_ref = base_sum(a_flat * b_flat);

    REQUIRE(c_neml2.base_dim() == 0); // scalar output
    REQUIRE(c_ref.base_dim() == 0);
    REQUIRE(c_neml2.dynamic_sizes() ==
            utils::broadcast_sizes(shape1.dynamic_sizes, shape2.dynamic_sizes));
    REQUIRE(c_neml2.intmd_sizes() ==
            utils::broadcast_sizes(shape1.intmd_sizes, shape2.intmd_sizes));

    REQUIRE_THAT(c_neml2, test::allclose(c_ref));
  }
}
