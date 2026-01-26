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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/mv.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEST_CASE("mv", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();
  auto shape1 = test::generate_tensor_shape<Scalar>();
  auto shape2 = test::generate_tensor_shape<Scalar>();
  auto m = GENERATE(1, 2);
  auto n = GENERATE(1, 3, 5);
  DYNAMIC_SECTION(cfg.desc() << " " << shape1.desc() << " " << shape2.desc() << " m: " << m
                             << " n: " << n)
  {
    auto a = test::generate_random_tensor<Tensor>(
        cfg, {shape1.dynamic_sizes, shape1.intmd_sizes, {m, n}});
    auto b =
        test::generate_random_tensor<Tensor>(cfg, {shape2.dynamic_sizes, shape2.intmd_sizes, {n}});
    auto c = neml2::mv(a, b);
    REQUIRE(c.dynamic_sizes() ==
            utils::broadcast_sizes(shape1.dynamic_sizes, shape2.dynamic_sizes));
    REQUIRE(c.intmd_sizes() == utils::broadcast_sizes(shape1.intmd_sizes, shape2.intmd_sizes));
    REQUIRE(c.base_sizes() == TensorShapeRef{m});
  }
}
