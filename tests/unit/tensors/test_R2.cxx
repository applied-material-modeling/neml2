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

#include <ATen/Context.h>
#include <catch2/catch_test_macros.hpp>

#include "neml2/misc/types.h"
#include "unit/tensors/generators.h"
#include "utils.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("R2", "[tensors]")
{
  at::manual_seed(42);

  SECTION("constructor")
  {
    SECTION("SR2")
    {
      auto S = R2::fill(1.1, 1.2, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3);
      auto s = SR2::fill(1.1, 2.2, 3.3, 2.3, 1.3, 1.2);
      REQUIRE_THAT(R2(s), test::allclose(S));
    }

    SECTION("WR2")
    {
      auto W = R2::fill(0.0, -1.2, 0.8, 1.2, 0.0, -0.5, -0.8, 0.5, 0.0);
      auto w = WR2::fill(0.5, 0.8, 1.2);
      REQUIRE_THAT(R2(w), test::allclose(W));
    }
  }

  SECTION("fill")
  {
    auto cfg = test::generate_tensor_config();
    DYNAMIC_SECTION(cfg.desc())
    {
      auto a = R2::fill(1, 2, 3, cfg.options);
      auto a0 = Tensor::create({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, cfg.options);
      REQUIRE_THAT(a, test::allclose(a0));

      auto a1 = Scalar::full({2, 3}, {4}, 1, cfg.options);
      auto a2 = Scalar::full({2, 3}, {4}, 2, cfg.options);
      auto a3 = Scalar::full({2, 3}, {4}, 3, cfg.options);
      a = R2::fill(a1, a2, a3);
      a0 = a0.batch_expand({2, 3}, {4});
      REQUIRE(a.dynamic_sizes() == TensorShapeRef{2, 3});
      REQUIRE(a.intmd_sizes() == TensorShapeRef{4});
      REQUIRE_THAT(a, test::allclose(a0));

      auto b = R2::fill(1, 2, 3, 4, 5, 6, cfg.options);
      auto b0 = Tensor::create({{1, 6, 5}, {6, 2, 4}, {5, 4, 3}}, cfg.options);
      REQUIRE_THAT(b, test::allclose(b0));

      auto b11 = Scalar::full({2, 1, 2}, {3}, 1, cfg.options);
      auto b22 = Scalar::full({2, 1, 2}, {3}, 2, cfg.options);
      auto b33 = Scalar::full({2, 1, 2}, {3}, 3, cfg.options);
      auto b23 = Scalar::full({2, 1, 2}, {3}, 4, cfg.options);
      auto b13 = Scalar::full({2, 1, 2}, {3}, 5, cfg.options);
      auto b12 = Scalar::full({2, 1, 2}, {3}, 6, cfg.options);
      b = R2::fill(b11, b22, b33, b23, b13, b12);
      b0 = b0.batch_expand({2, 1, 2}, {3});
      REQUIRE(b.dynamic_sizes() == TensorShapeRef{2, 1, 2});
      REQUIRE(b.intmd_sizes() == TensorShapeRef{3});
      REQUIRE_THAT(b, test::allclose(b0));
    }
  }

  SECTION("skew")
  {
    auto cfg = test::generate_tensor_config();
    DYNAMIC_SECTION(cfg.desc())
    {
      auto v = Vec::fill(1, 2, 3, cfg.options);
      auto s = R2::skew(v);
      auto s0 = Tensor::create({{0, -3, 2}, {3, 0, -1}, {-2, 1, 0}}, cfg.options);
      REQUIRE_THAT(s, test::allclose(s0));
    }
  }

  SECTION("identity")
  {
    auto cfg = test::generate_tensor_config();
    DYNAMIC_SECTION(cfg.desc())
    {
      auto I = R2::identity(cfg.options);
      auto I0 = Tensor::create({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, cfg.options);
      REQUIRE_THAT(I, test::allclose(I0));
    }
  }

  SECTION("rotate")
  {
    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto T = R2::fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    auto Tp = R2::fill(
        -1.02332, -0.0592151, -0.290549, 0.440785, 0.208734, -1.65399, -5.14556, -2.0769, 15.8146);

    REQUIRE_THAT(T.rotate(r), test::allclose(Tp));

    auto apply = [T](const Tensor & x) { return T.rotate(Rot(x)); };
    auto dTp_dr = finite_differencing_derivative(apply, r);
    REQUIRE_THAT(T.drotate(r), test::allclose(dTp_dr));

    auto R = r.euler_rodrigues();
    auto apply_R = [T](const Tensor & x) { return T.rotate(R2(x)); };
    auto dTp_dR = finite_differencing_derivative(apply_R, R);
    REQUIRE_THAT(T.drotate(R), test::allclose(dTp_dR));
  }

  SECTION("row")
  {
    auto a = R2::rand({2, 3}, {2, 1});
    for (Size i = 0; i < 3; i++)
    {
      auto ar = a.row(i);
      auto ar0 = a.index({indexing::Ellipsis, i, indexing::Slice()});
      REQUIRE_THAT(ar, test::allclose(ar0));
    }
  }

  SECTION("col")
  {
    auto a = R2::rand({2, 3}, {2, 1});
    for (Size i = 0; i < 3; i++)
    {
      auto ac = a.col(i);
      auto ac0 = a.index({indexing::Ellipsis, indexing::Slice(), i});
      REQUIRE_THAT(ac, test::allclose(ac0));
    }
  }

  SECTION("transpose")
  {
    auto a = R2::rand({2, 3}, {2, 1});
    auto at = a.transpose();
    REQUIRE_THAT(at(0, 0), test::allclose(a(0, 0)));
    REQUIRE_THAT(at(1, 1), test::allclose(a(1, 1)));
    REQUIRE_THAT(at(2, 2), test::allclose(a(2, 2)));
    REQUIRE_THAT(at(1, 2), test::allclose(a(2, 1)));
    REQUIRE_THAT(at(0, 2), test::allclose(a(2, 0)));
    REQUIRE_THAT(at(0, 1), test::allclose(a(1, 0)));
  }
}
