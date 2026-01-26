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

#include <functional>

#include "neml2/tensors/shape_utils.h"
#include "unit/tensors/generators.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEMPLATE_TEST_CASE("operators",
                   "[tensors/functions]",
                   std::plus<>,
                   std::minus<>,
                   std::multiplies<>,
                   std::divides<>)
{
  auto op = TestType{};
  auto cfg = test::generate_tensor_config();

  SECTION("Tensor, Tensor")
  {
    auto shape1 = test::generate_tensor_shape();
    auto shape2 = test::generate_tensor_shape();

    DYNAMIC_SECTION(cfg.desc() << " LHS: " << shape1.desc() << " RHS: " << shape2.desc())
    {
      // Our implementation
      auto a = test::generate_full_tensor<Tensor>(cfg, shape1, 2.0);
      auto b = test::generate_full_tensor<Tensor>(cfg, shape2, 1.0);
      auto c = op(a, b);

      // Reference implementation
      auto ds = utils::broadcast_sizes(shape1.dynamic_sizes, shape2.dynamic_sizes);
      auto is = utils::broadcast_sizes(shape1.intmd_sizes, shape2.intmd_sizes);
      auto bs = utils::broadcast_sizes(shape1.base_sizes, shape2.base_sizes);
      auto shape = test::GeneratedTensorShape(ds, is, bs);
      auto dtype = c10::isIntegralType(cfg.dtype, /*includeBool=*/false)
                       ? (std::is_same_v<TestType, std::divides<>>
                              ? get_default_dtype()
                              : c10::promoteTypes(a.scalar_type(), b.scalar_type()))
                       : c10::promoteTypes(a.scalar_type(), b.scalar_type());
      auto ccfg = test::GeneratedTensorConfig(dtype, cfg.device);
      auto cc = test::generate_full_tensor<Tensor>(ccfg, shape, op(2.0, 1.0));

      REQUIRE(test::match_tensor_config(c, ccfg));
      REQUIRE(test::match_tensor_shape(c, shape));
      REQUIRE_THAT(c, test::allclose(cc));
    }
  }

  SECTION("Tensor, Scalar")
  {
    auto shape1 = test::generate_tensor_shape();
    auto shape2 = test::generate_tensor_shape<Scalar>();

    DYNAMIC_SECTION(cfg.desc() << " LHS: " << shape1.desc() << " RHS: " << shape2.desc())
    {
      // Our implementation
      auto a = test::generate_full_tensor<Tensor>(cfg, shape1, 2.0);
      auto b = test::generate_full_tensor<Scalar>(cfg, shape2, 1.0);
      auto c = op(a, b);

      // Reference implementation
      auto ds = utils::broadcast_sizes(shape1.dynamic_sizes, shape2.dynamic_sizes);
      auto is = utils::broadcast_sizes(shape1.intmd_sizes, shape2.intmd_sizes);
      auto bs = shape1.base_sizes;
      auto shape = test::GeneratedTensorShape(ds, is, bs);
      auto dtype = c10::isIntegralType(cfg.dtype, /*includeBool=*/false)
                       ? (std::is_same_v<TestType, std::divides<>>
                              ? get_default_dtype()
                              : c10::promoteTypes(a.scalar_type(), b.scalar_type()))
                       : c10::promoteTypes(a.scalar_type(), b.scalar_type());
      auto ccfg = test::GeneratedTensorConfig(dtype, cfg.device);
      auto cc = test::generate_full_tensor<Tensor>(ccfg, shape, op(2.0, 1.0));

      REQUIRE(test::match_tensor_config(c, ccfg));
      REQUIRE(test::match_tensor_shape(c, shape));
      REQUIRE_THAT(c, test::allclose(cc));
    }
  }

  SECTION("Tensor, CScalar")
  {
    auto shape = test::generate_tensor_shape();

    DYNAMIC_SECTION(cfg.desc() << " LHS: " << shape.desc())
    {
      // Our implementation
      auto a = test::generate_full_tensor<Tensor>(cfg, shape, 2.0);
      auto b = CScalar(1);
      auto c = op(a, b);

      // Reference implementation
      auto dtype =
          c10::isIntegralType(cfg.dtype, /*includeBool=*/false)
              ? (std::is_same_v<TestType, std::divides<>> ? get_default_dtype() : a.scalar_type())
              : a.scalar_type();
      auto ccfg = test::GeneratedTensorConfig(dtype, cfg.device);
      auto cc = test::generate_full_tensor<Tensor>(ccfg, shape, op(2.0, 1.0));

      REQUIRE(test::match_tensor_config(c, ccfg));
      REQUIRE(test::match_tensor_shape(c, shape));
      REQUIRE_THAT(c, test::allclose(cc));
    }
  }
}

TEST_CASE("in-place operators", "[tensors/functions]")
{
  auto cfg = test::generate_tensor_config(test::fp_dtypes());
  auto shape = test::GeneratedTensorShape({4, 3}, {2}, {5});

  DYNAMIC_SECTION(cfg.desc())
  {
    auto a = test::generate_full_tensor<Tensor>(cfg, shape, 2.0);
    a += 2;
    REQUIRE_THAT(a, test::allclose(at::full_like(a, 4)));
    a *= 2;
    REQUIRE_THAT(a, test::allclose(at::full_like(a, 8)));
    a -= 2;
    REQUIRE_THAT(a, test::allclose(at::full_like(a, 6)));
    a /= 2;
    REQUIRE_THAT(a, test::allclose(at::full_like(a, 3)));
  }
}
