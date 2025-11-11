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

#include <c10/core/ScalarType.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "neml2/tensors/functions/det.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/WR2.h"
#include "neml2/tensors/functions/symmetrization.h"
#include "neml2/tensors/shape_utils.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("det", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();

  SECTION("R2")
  {
    auto shape = test::generate_tensor_shape<R2>();
    auto scalar_shape = test::GeneratedTensorShape(
          shape.dynamic_sizes, shape.intmd_sizes, {});
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
      auto A = test::generate_random_tensor<R2>(cfg, shape);
      auto detA = neml2::det(A);

      auto a = A(0, 0);
      auto b = A(0, 1);
      auto c = A(0, 2);
      auto d = A(1, 0);
      auto e = A(1, 1);
      auto f = A(1, 2);
      auto g = A(2, 0);
      auto h = A(2, 1);
      auto i = A(2, 2);

      auto ref = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

      REQUIRE(test::match_tensor_shape(detA, scalar_shape));
      REQUIRE_THAT(detA, test::allclose(ref));
    }
  }

  SECTION("SR2")
  {
    auto shape = test::generate_tensor_shape<SR2>();
    auto scalar_shape = test::GeneratedTensorShape(
          shape.dynamic_sizes, shape.intmd_sizes, {});
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
        auto A = test::generate_random_tensor<SR2>(cfg, shape);
        auto detA = neml2::det(A);

        const auto comps = at::split(A.base_flatten(), 1, -1);
        const auto & a = comps[0];
        const auto & e = comps[1];
        const auto & i = comps[2];
        const auto f = comps[3] / mandel_factor(3);
        const auto c = comps[4] / mandel_factor(4);
        const auto b = comps[5] / mandel_factor(5);

        auto ref = a * (e * i - f * f) - b * (b * i - c * f) + c * (b * f - e * c);
        
        REQUIRE(test::match_tensor_shape(detA, scalar_shape));

        auto ref_flat = ref.reshape_as(detA);
        REQUIRE_THAT(detA, test::allclose(ref_flat));
    }
  }

  SECTION("WR2")
  {
    auto shape = test::generate_tensor_shape<WR2>();
    auto scalar_shape = test::GeneratedTensorShape(
          shape.dynamic_sizes, shape.intmd_sizes, {});
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
      auto A = test::generate_random_tensor<WR2>(cfg, shape);
      auto detA = neml2::det(A);

      REQUIRE(test::match_tensor_shape(detA, scalar_shape));
      REQUIRE_THAT(detA, test::allclose(Scalar::zeros_like(detA)));
    }
  }
}
