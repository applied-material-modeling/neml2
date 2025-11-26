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
#include <catch2/generators/catch_generators.hpp>
#include <optional>

#include <ATen/ops/tensor.h>

#include "neml2/tensors/tensors.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

#define TYPE_IDENTITY(T) T

TEMPLATE_TEST_CASE("PrimitiveTensor", "[tensors]", FOR_ALL_PRIMITIVETENSOR_COMMA(TYPE_IDENTITY))
{
  at::manual_seed(42);

  SECTION("constants")
  {
    auto a = TestType::empty();
    REQUIRE(TestType::const_base_sizes == a.base_sizes());
    REQUIRE(TestType::const_base_dim == a.base_dim());
    REQUIRE(TestType::const_base_numel == utils::numel(a.base_sizes()));
  }

  SECTION("constructors")
  {
    SECTION("default")
    {
      auto a = TestType();
      REQUIRE(!a.defined());
    }

    SECTION("ATensor, intermediate dimension")
    {
      auto cfg = test::generate_tensor_config();
      auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});
      DYNAMIC_SECTION(cfg.desc())
      {
        auto a = test::generate_random_tensor(cfg, shape);
        auto b = TestType(a, shape.intmd_dim);
        REQUIRE(test::match_tensor_config(b, cfg));
        REQUIRE(test::match_tensor_shape(b, shape));
        REQUIRE_THAT(a, test::allclose(ATensor(b)));
      }
    }

    SECTION("copy")
    {
      auto cfg = test::generate_tensor_config();
      auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});
      DYNAMIC_SECTION(cfg.desc())
      {
        auto a = test::generate_random_tensor<Tensor>(cfg, shape);
        auto b = TestType(a);
        REQUIRE(test::match_tensor_config(b, cfg));
        REQUIRE(test::match_tensor_shape(b, shape));
        REQUIRE_THAT(a, test::allclose(b));
      }
    }

    SECTION("operator neml2::Tensor")
    {
      auto cfg = test::generate_tensor_config();
      auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});
      DYNAMIC_SECTION(cfg.desc())
      {
        auto a = test::generate_random_tensor<Tensor>(cfg, shape);
        TestType b(a);
        neml2::Tensor c = b;
        REQUIRE(test::match_tensor_config(c, cfg));
        REQUIRE(test::match_tensor_shape(c, shape));
        REQUIRE_THAT(a, test::allclose(c));
      }
    }
  }

  SECTION("empty")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});

    auto a = TestType::empty(cfg.options);
    REQUIRE(test::match_tensor_config(a, cfg));
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{});
    REQUIRE(a.base_sizes() == TestType::const_base_sizes);

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a = TestType::empty(shape.dynamic_sizes, shape.intmd_sizes, cfg.options);
      REQUIRE(test::match_tensor_config(a, cfg));
      REQUIRE(test::match_tensor_shape(a, shape));
    }
  }

  SECTION("zeros")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});

    auto a0 = at::zeros(TestType::const_base_sizes, cfg.options);
    auto a = TestType::zeros(cfg.options);
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{});
    REQUIRE(a.base_sizes() == TestType::const_base_sizes);
    REQUIRE_THAT(a, test::allclose(a0));

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a0 = test::generate_full_tensor(cfg, shape, 0);
      auto a = TestType::zeros(shape.dynamic_sizes, shape.intmd_sizes, cfg.options);
      REQUIRE(test::match_tensor_shape(a, shape));
      REQUIRE_THAT(a, test::allclose(a0));
    }
  }

  SECTION("ones")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});

    auto a0 = at::ones(TestType::const_base_sizes, cfg.options);
    auto a = TestType::ones(cfg.options);
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{});
    REQUIRE(a.base_sizes() == TestType::const_base_sizes);
    REQUIRE_THAT(a, test::allclose(a0));

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a0 = test::generate_full_tensor(cfg, shape, 1);
      auto a = TestType::ones(shape.dynamic_sizes, shape.intmd_sizes, cfg.options);
      REQUIRE(test::match_tensor_shape(a, shape));
      REQUIRE_THAT(a, test::allclose(a0));
    }
  }

  SECTION("full")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});

    auto a0 = at::full(TestType::const_base_sizes, 7.5, cfg.options);
    auto a = TestType::full(7.5, cfg.options);
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{});
    REQUIRE(a.base_sizes() == TestType::const_base_sizes);
    REQUIRE_THAT(a, test::allclose(a0));

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a0 = test::generate_full_tensor(cfg, shape, 7.5);
      auto a = TestType::full(shape.dynamic_sizes, shape.intmd_sizes, 7.5, cfg.options);
      REQUIRE(test::match_tensor_shape(a, shape));
      REQUIRE_THAT(a, test::allclose(a0));
    }
  }

  SECTION("rand")
  {
    auto cfg = test::generate_tensor_config(test::fp_dtypes());
    auto shape = test::GeneratedTensorShape({}, {}, {TestType::const_base_sizes});

    auto a0 = at::rand(TestType::const_base_sizes, cfg.options);
    auto a = TestType::rand(cfg.options);
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{});
    REQUIRE(a.base_sizes() == TestType::const_base_sizes);

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a0 = test::generate_random_tensor(cfg, shape);
      auto a = TestType::rand(shape.dynamic_sizes, shape.intmd_sizes, cfg.options);
      REQUIRE(test::match_tensor_shape(a, shape));
    }
  }
}

TEST_CASE("PrimitiveTensor", "[tensors]")
{
  SECTION("create")
  {
    auto cfg = test::GeneratedTensorConfig{kFloat64, kCPU};
    auto a = Vec::create({{{1, 2, 3}, {2, 3, 4}},
                          {{2, 3, 4}, {3, 4, 5}},
                          {{3, 4, 5}, {4, 5, 6}},
                          {{4, 5, 6}, {5, 6, 7}}},
                         1,
                         cfg.options);
    auto a0 = at::tensor({1, 2, 3, 2, 3, 4, 2, 3, 4, 3, 4, 5, 3, 4, 5, 4, 5, 6, 4, 5, 6, 5, 6, 7},
                         cfg.options)
                  .reshape({4, 2, 3});
    REQUIRE(a.dynamic_sizes() == TensorShapeRef{4});
    REQUIRE(a.intmd_sizes() == TensorShapeRef{2});
    REQUIRE(a.base_sizes() == TensorShapeRef{3});
    REQUIRE_THAT(a, test::allclose(a0));
  }

  SECTION("fill")
  {
    auto cfg = test::generate_tensor_config();
    DYNAMIC_SECTION(cfg.desc())
    {
      auto a = Vec::fill(1, 2, 3, cfg.options);
      auto a0 = at::linspace(1, 3, 3, cfg.options);
      REQUIRE_THAT(a, test::allclose(a0));

      auto a1 = Scalar::full({2, 3}, {4}, 1, cfg.options);
      auto a2 = Scalar::full({2, 3}, {4}, 2, cfg.options);
      auto a3 = Scalar::full({2, 3}, {4}, 3, cfg.options);
      a = Vec::fill(a1, a2, a3);
      a0 = a0.expand({2, 3, 4, 3});
      REQUIRE(a.dynamic_sizes() == TensorShapeRef{2, 3});
      REQUIRE(a.intmd_sizes() == TensorShapeRef{4});
      REQUIRE_THAT(a, test::allclose(a0));

      auto b = R2::fill(1, 2, 3, 4, 5, 6, 7, 8, 9, cfg.options);
      auto b0 = at::linspace(1, 9, 9, cfg.options).reshape({3, 3});
      REQUIRE_THAT(b, test::allclose(b0));

      auto b11 = Scalar::full({2, 1, 2}, {3}, 1, cfg.options);
      auto b12 = Scalar::full({2, 1, 2}, {3}, 2, cfg.options);
      auto b13 = Scalar::full({2, 1, 2}, {3}, 3, cfg.options);
      auto b21 = Scalar::full({2, 1, 2}, {3}, 4, cfg.options);
      auto b22 = Scalar::full({2, 1, 2}, {3}, 5, cfg.options);
      auto b23 = Scalar::full({2, 1, 2}, {3}, 6, cfg.options);
      auto b31 = Scalar::full({2, 1, 2}, {3}, 7, cfg.options);
      auto b32 = Scalar::full({2, 1, 2}, {3}, 8, cfg.options);
      auto b33 = Scalar::full({2, 1, 2}, {3}, 9, cfg.options);
      b = R2::fill(b11, b12, b13, b21, b22, b23, b31, b32, b33);
      b0 = b0.expand({2, 1, 2, 3, 3, 3});
      REQUIRE(b.dynamic_sizes() == TensorShapeRef{2, 1, 2});
      REQUIRE(b.intmd_sizes() == TensorShapeRef{3});
      REQUIRE_THAT(b, test::allclose(b0));
    }
  }

  SECTION("einsum")
  {
    auto cfg = test::generate_tensor_config();
    auto shape_a = test::generate_tensor_shape<R3>();
    auto shape_b = test::generate_tensor_shape<R4>();
    DYNAMIC_SECTION(cfg.desc() << " " << shape_a.desc() << " " << shape_b.desc())
    {
      auto a = test::generate_random_tensor<R3>(cfg, shape_a);
      auto b = test::generate_random_tensor<R4>(cfg, shape_b);
      auto c = Vec::einsum("...ijk,...kijm->...m", {a, b});

      auto [a0, b0, i] = utils::align_intmd_dim(a, b);
      auto c0 = at::einsum("...ijk,...kijm->...m", {a0, b0});

      REQUIRE_THAT(c, test::allclose(c0));
    }
  }

  SECTION("operator()")
  {
    auto cfg = test::generate_tensor_config();
    auto dynamic_sizes = GENERATE_REF(TensorShape{}, TensorShape{2, 1, 3});
    auto intmd_sizes = GENERATE_REF(TensorShape{}, TensorShape{4});
    DYNAMIC_SECTION(cfg.desc())
    {
      auto a = test::generate_random_tensor<Vec>(
          cfg, {dynamic_sizes, intmd_sizes, Vec::const_base_sizes});
      for (Size i = 0; i < 3; i++)
        REQUIRE_THAT(a(i), test::allclose(a.base_index({i})));

      auto b =
          test::generate_random_tensor<R3>(cfg, {dynamic_sizes, intmd_sizes, R3::const_base_sizes});
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
            REQUIRE_THAT(b(i, j, k), test::allclose(b.base_index({i, j, k})));
    }
  }
}
