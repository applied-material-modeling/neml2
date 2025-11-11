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
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/linalg/lu_factor.h"
#include "neml2/tensors/functions/linalg/lu_solve.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("lu - same batch shapes", "[tensors/functions/linag]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(std::vector<c10::ScalarType>({kFloat32, kFloat64}));

  auto db = std::vector<TensorShape>({{}, {2, 1}});
  auto ib = std::vector<TensorShape>({{}, {1, 3}});
  auto bs = std::vector<TensorShape>({{3, 3}});
  auto shape = test::generate_tensor_shape<Tensor>(db, ib, bs);

  DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
  {
    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto vec_shape = test::GeneratedTensorShape(shape.dynamic_sizes, shape.intmd_sizes, {3, 1});
    auto b = test::generate_random_tensor<Tensor>(cfg, vec_shape);

    auto [lu, pivots] = neml2::linalg::lu_factor(a);
    auto x = neml2::linalg::lu_solve(lu, pivots, b);

    REQUIRE(test::match_tensor_shape(x, vec_shape));
    REQUIRE_THAT(neml2::mm(a, x), test::allclose_broadcast(b));
  }
}

TEST_CASE("lu - broadcastable batch shapes", "[tensors/functions/linag]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config(std::vector<c10::ScalarType>({kFloat32, kFloat64}));

  auto db = std::vector<TensorShape>({{}, {2, 1}});
  auto ib = std::vector<TensorShape>({{}, {1, 3}});
  auto bs = std::vector<TensorShape>({{3, 3}});
  auto vs = std::vector<TensorShape>({{3, 1}});
  auto shape1 = test::generate_tensor_shape<Tensor>(db, ib, bs);
  auto shape2 = test::generate_tensor_shape<Tensor>(db, ib, vs);

  DYNAMIC_SECTION(cfg.desc() << " " << shape1.desc() << " " << shape2.desc())
  {
    auto a = test::generate_random_tensor<Tensor>(cfg, shape1);
    auto b = test::generate_random_tensor<Tensor>(cfg, shape2);

    auto [lu, pivots] = neml2::linalg::lu_factor(a);
    auto x = neml2::linalg::lu_solve(lu, pivots, b);

    REQUIRE_THAT(neml2::mm(a, x), test::allclose_broadcast(b));
  }
}
