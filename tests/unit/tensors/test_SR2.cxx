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

#include "unit/tensors/generators.h"
#include "utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/diagonalize.h"

using namespace neml2;

TEST_CASE("SR2", "[tensors]")
{
  at::manual_seed(42);

  SECTION("constructors")
  {
    SECTION("R2")
    {
      auto S = R2::fill(1.1, 1.2, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3);
      auto s = SR2::fill(1.1, 2.2, 3.3, 2.3, 1.3, 1.2);
      REQUIRE(at::allclose(SR2(S), s));
    }
  }

  SECTION("fill")
  {
    auto cfg = test::generate_tensor_config({{kFloat32, kFloat64}});

    DYNAMIC_SECTION(cfg.desc())
    {
      auto a = SR2::fill(2, cfg.options);
      auto a0 = at::tensor({2., 2., 2., 0., 0., 0.}, cfg.options);
      REQUIRE_THAT(a, test::allclose(a0));

      auto b = SR2::fill(1, 2, 3, cfg.options);
      auto b0 = at::tensor({1., 2., 3., 0., 0., 0.}, cfg.options);
      REQUIRE_THAT(b, test::allclose(b0));

      auto c = SR2::fill(1, 2, 3, 4, 5, 6, cfg.options);
      auto c0 = at::tensor({1., 2., 3., 4 * sqrt2, 5 * sqrt2, 6 * sqrt2}, cfg.options);
      REQUIRE_THAT(c, test::allclose(c0));
    }

    auto shape = test::generate_tensor_shape<Scalar>();
    DYNAMIC_SECTION(cfg.desc() + " " + shape.desc())
    {
      auto a = test::generate_random_tensor<Scalar>(cfg, shape);
      auto A = SR2::fill(a);
      auto A0 = a * Tensor::identity(3, cfg.options);
      REQUIRE_THAT(R2(A), test::allclose(A0));

      auto b1 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto b2 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto b3 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto B = SR2::fill(b1, b2, b3);
      auto B0 = base_diagonalize(base_stack({b1, b2, b3}));
      REQUIRE_THAT(R2(B), test::allclose(B0));

      auto c1 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto c2 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto c3 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto c4 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto c5 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto c6 = test::generate_random_tensor<Scalar>(cfg, shape);
      auto C = SR2::fill(c1, c2, c3, c4, c5, c6);
      auto C1 = base_stack({c1, c6, c5});
      auto C2 = base_stack({c6, c2, c4});
      auto C3 = base_stack({c5, c4, c3});
      auto C0 = base_stack({C1, C2, C3});
      REQUIRE_THAT(R2(C), test::allclose(C0));
    }
  }

  SECTION("identity")
  {
    auto a = SR2::identity();
    auto b = at::eye(3);
    REQUIRE(at::allclose(R2(a), b));
  }

  SECTION("rotate")
  {
    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto T = SR2(R2::fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    auto Tp = SR2::fill(-1.02332, 0.208734, 15.8146, -1.86545, -2.71806, 0.190785);

    REQUIRE(at::allclose(T.rotate(r), Tp));

    // rotate by MRP
    auto apply_r = [T](const Tensor & x) { return T.rotate(Rot(x)); };
    auto dTp_dr = finite_differencing_derivative(apply_r, r);
    REQUIRE(at::allclose(T.drotate(r), dTp_dr, 1e-4));

    // rotate by rotation matrix
    auto R = r.euler_rodrigues();
    auto apply_R = [T](const Tensor & x) { return T.rotate(R2(x)); };
    auto dTp_dR = finite_differencing_derivative(apply_R, R);
    REQUIRE(at::allclose(T.drotate(R), dTp_dR));
  }

  SECTION("operator()")
  {
    auto a = SR2::rand({}, {});
    auto b = R2(a);
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        REQUIRE_THAT(a(i, j), test::allclose(b(i, j)));
  }

  SECTION("transpose")
  {
    // no-op
  }
}
