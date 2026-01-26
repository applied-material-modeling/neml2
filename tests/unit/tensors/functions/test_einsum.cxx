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

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/einsum.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("einsum", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape1 = test::GeneratedTensorShape{{2, 1}, {4}, {}};
  auto shape2 = test::GeneratedTensorShape{{3, 2, 2}, {}, {3, 3}};
  auto shape3 = test::GeneratedTensorShape{{1, 2}, {2, 1}, {3, 3, 3}};
  DYNAMIC_SECTION(cfg.desc())
  {
    auto a1 = test::generate_random_tensor<Tensor>(cfg, shape1);
    auto a2 = test::generate_random_tensor<Tensor>(cfg, shape2);
    auto a3 = test::generate_random_tensor<Tensor>(cfg, shape3);
    auto b = neml2::einsum("...,...ik,...lki", {a1, a2, a3});
    auto b0 =
        at::einsum("...,...ik,...lki",
                   {a1.intmd_unsqueeze(0, 1), a2.intmd_unsqueeze(0, 2), a3.intmd_unsqueeze(0, 0)});
    REQUIRE(b.dynamic_sizes() == TensorShapeRef{3, 2, 2});
    REQUIRE(b.intmd_sizes() == TensorShapeRef{2, 4});
    REQUIRE(b.base_sizes() == TensorShapeRef{3});
    REQUIRE_THAT(b, test::allclose(b0));
  }
}
