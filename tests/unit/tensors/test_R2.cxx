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

#include "utils.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("R2", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class R2")
  {
    SECTION("R2")
    {
      SECTION("from SR2")
      {
        auto S = R2::fill(1.1, 1.2, 1.3, 1.2, 2.2, 2.3, 1.3, 2.3, 3.3);
        auto s = SR2::fill(1.1, 2.2, 3.3, 2.3, 1.3, 1.2);
        REQUIRE(at::allclose(R2(s), S));
      }
      SECTION("from WR2")
      {
        auto W = R2::fill(0, -1.2, 0.8, 1.2, 0, -0.5, -0.8, 0.5, 0);
        auto w = WR2::fill(0.5, 0.8, 1.2);
        REQUIRE(at::allclose(R2(w), W));
      }
      SECTION("from Rot")
      {
        auto r = Rot(at::rand(utils::add_shapes(B, 3), DTO));
        REQUIRE(at::allclose(R2(r), r.euler_rodrigues()));
      }
    }

    SECTION("fill")
    {
      SECTION("fill from 1 value")
      {
        auto a1 = R2::fill(1.1, DTO);
        auto a2 = R2::fill(Scalar(1.1, DTO));
        auto a3 = R2::fill(Scalar::full(B, 1.1, DTO));
        auto b = R2::create({{1.1, 0.0, 0.0}, {0.0, 1.1, 0.0}, {0.0, 0.0, 1.1}}, DTO);
        REQUIRE(at::allclose(a1, b));
        REQUIRE(at::allclose(a2, b));
        REQUIRE(at::allclose(a3, b.batch_expand(B)));
      }
      SECTION("fill from 3 values")
      {
        auto a1 = R2::fill(1.1, 2.2, 3.3, DTO);
        auto a2 = R2::fill(Scalar(1.1, DTO), Scalar(2.2, DTO), Scalar(3.3, DTO));
        auto a3 = R2::fill(
            Scalar::full(B, 1.1, DTO), Scalar::full(B, 2.2, DTO), Scalar::full(B, 3.3, DTO));
        auto b = R2::create({{1.1, 0.0, 0.0}, {0.0, 2.2, 0.0}, {0.0, 0.0, 3.3}}, DTO);
        REQUIRE(at::allclose(a1, b));
        REQUIRE(at::allclose(a2, b));
        REQUIRE(at::allclose(a3, b.batch_expand(B)));
      }
      SECTION("fill from 6 values")
      {
        auto a1 = R2::fill(1.1, 2.2, 3.3, 2.3, 1.3, 1.2, DTO);
        auto a2 = R2::fill(Scalar(1.1, DTO),
                           Scalar(2.2, DTO),
                           Scalar(3.3, DTO),
                           Scalar(2.3, DTO),
                           Scalar(1.3, DTO),
                           Scalar(1.2, DTO));
        auto a3 = R2::fill(Scalar::full(B, 1.1, DTO),
                           Scalar::full(B, 2.2, DTO),
                           Scalar::full(B, 3.3, DTO),
                           Scalar::full(B, 2.3, DTO),
                           Scalar::full(B, 1.3, DTO),
                           Scalar::full(B, 1.2, DTO));
        auto b = R2::create({{1.1, 1.2, 1.3}, {1.2, 2.2, 2.3}, {1.3, 2.3, 3.3}}, DTO);
        REQUIRE(at::allclose(a1, b));
        REQUIRE(at::allclose(a2, b));
        REQUIRE(at::allclose(a3, b.batch_expand(B)));
      }
      SECTION("fill from 9 values")
      {
        auto a1 = R2::fill(1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, DTO);
        auto a2 = R2::fill(Scalar(1.1, DTO),
                           Scalar(1.2, DTO),
                           Scalar(1.3, DTO),
                           Scalar(2.1, DTO),
                           Scalar(2.2, DTO),
                           Scalar(2.3, DTO),
                           Scalar(3.1, DTO),
                           Scalar(3.2, DTO),
                           Scalar(3.3, DTO));
        auto a3 = R2::fill(Scalar::full(B, 1.1, DTO),
                           Scalar::full(B, 1.2, DTO),
                           Scalar::full(B, 1.3, DTO),
                           Scalar::full(B, 2.1, DTO),
                           Scalar::full(B, 2.2, DTO),
                           Scalar::full(B, 2.3, DTO),
                           Scalar::full(B, 3.1, DTO),
                           Scalar::full(B, 3.2, DTO),
                           Scalar::full(B, 3.3, DTO));
        auto b = R2::create({{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}, DTO);
        REQUIRE(at::allclose(a1, b));
        REQUIRE(at::allclose(a2, b));
        REQUIRE(at::allclose(a3, b.batch_expand(B)));
      }
    }

    SECTION("skew")
    {
      auto a = Vec::fill(1.2, 2.1, -1.5, DTO);
      auto b = R2::skew(a);
      auto c = R2::skew(a.batch_expand(B));
      REQUIRE(at::allclose(b.transpose(), -b));
      REQUIRE(at::allclose(c.transpose(), -c));
    }

    SECTION("identity")
    {
      auto a = R2::identity(DTO);
      auto b = at::eye(3, DTO);
      REQUIRE(at::allclose(a, b));
    }

    SECTION("rotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto T = R2::fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
      auto Tp = R2::fill(-1.02332,
                         -0.0592151,
                         -0.290549,
                         0.440785,
                         0.208734,
                         -1.65399,
                         -5.14556,
                         -2.0769,
                         15.8146);

      auto rb = r.batch_expand(B);
      auto Tb = T.batch_expand(B);
      auto Tpb = Tp.batch_expand(B);

      REQUIRE(at::allclose(T.rotate(r), Tp));
      REQUIRE(at::allclose(Tb.rotate(rb), Tpb));
      REQUIRE(at::allclose(T.rotate(rb), Tpb));
      REQUIRE(at::allclose(Tb.rotate(r), Tpb));
    }

    SECTION("drotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto T = R2::fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
      auto Tp = R2::fill(-1.02332,
                         -0.0592151,
                         -0.290549,
                         0.440785,
                         0.208734,
                         -1.65399,
                         -5.14556,
                         -2.0769,
                         15.8146);

      auto rb = r.batch_expand(B);
      auto Tb = T.batch_expand(B);
      auto Tpb = Tp.batch_expand(B);

      auto apply = [T](const Tensor & x) { return T.rotate(Rot(x)); };
      auto dTp_dr = finite_differencing_derivative(apply, r);
      auto dTp_drb = dTp_dr.batch_expand(B);

      REQUIRE(at::allclose(T.drotate(r), dTp_dr));
      REQUIRE(at::allclose(Tb.drotate(rb), dTp_drb));
      REQUIRE(at::allclose(T.drotate(rb), dTp_drb));
      REQUIRE(at::allclose(Tb.drotate(r), dTp_drb));
    }

    SECTION("operator()")
    {
      auto a = R2(at::rand(utils::add_shapes(B, 3, 3), DTO));
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          REQUIRE(at::allclose(a(i, j), a.index({indexing::Ellipsis, i, j})));
    }

    SECTION("inverse")
    {
      auto a = R2(at::rand({3, 3}, DTO));
      auto ai = R2(ATensor(a).inverse());
      REQUIRE(at::allclose(a.inverse(), ai));
      REQUIRE(at::allclose(a.batch_expand(B).inverse(), ai.batch_expand(B)));
    }

    SECTION("transpose")
    {
      auto a = R2::fill(1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, DTO);
      auto at = R2::fill(1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 1.3, 2.3, 3.3, DTO);
      REQUIRE(at::allclose(a.transpose(), at));
      REQUIRE(at::allclose(a.batch_expand(B).transpose(), at.batch_expand(B)));
    }
  }

  SECTION("operator*")
  {
    auto a = R2::create({{0.89072077, 0.82632195, 0.04234413},
                         {0.85614465, 0.9737414, 0.9491076},
                         {0.07230831, 0.49570559, 0.42566357}},
                        DTO);
    SECTION("R2 * R2")
    {
      auto b = R2::create({{0.94487948, 0.15144116, 0.03378262},
                           {0.51449125, 0.56508379, 0.06392479},
                           {0.46572483, 0.00443049, 0.42061576}},
                          DTO);
      auto c = R2::create({{1.28647989, 0.60202053, 0.10072394},
                           {1.75195791, 0.68410603, 0.49037864},
                           {0.52160092, 0.29295155, 0.21317145}},
                          DTO);
      REQUIRE(at::allclose(a * b, c));
      REQUIRE(at::allclose(a.batch_expand(B) * b, c.batch_expand(B)));
      REQUIRE(at::allclose(a * b.batch_expand(B), c.batch_expand(B)));
      REQUIRE(at::allclose(a.batch_expand(B) * b.batch_expand(B), c.batch_expand(B)));
    }
    SECTION("R2 * Vec")
    {
      auto b = Vec::fill(0.05425937, 0.55065082, 0.24673347);
      auto c = Vec::fill(0.51379251, 0.81682197, 0.38190954);
      REQUIRE(at::allclose(a * b, c));
      REQUIRE(at::allclose(a.batch_expand(B) * b, c.batch_expand(B)));
      REQUIRE(at::allclose(a * b.batch_expand(B), c.batch_expand(B)));
      REQUIRE(at::allclose(a.batch_expand(B) * b.batch_expand(B), c.batch_expand(B)));
    }
  }
}
