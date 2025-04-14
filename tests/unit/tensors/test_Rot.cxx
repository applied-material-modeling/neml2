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

TEST_CASE("Rot", "[tensors]")
{
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class Rot")
  {
    SECTION("identity")
    {
      auto a = Rot::identity(DTO);
      auto b = Vec(at::rand(utils::add_shapes(B, 3), DTO));
      // Rotate by an identity rotation should be a no-op
      REQUIRE(at::allclose(b.rotate(a), b));
    }

    SECTION("rotate_to")
    {
      auto va = Vec::fill(1.0, 2.0, 3.0, DTO);
      va /= va.norm();
      auto vb = Vec::fill(4.0, 5.0, 6.0, DTO);
      vb /= vb.norm();
      auto r = Rot::rotation_from_to(va, vb);
      REQUIRE(at::allclose(va.rotate(r), vb));
    }

    SECTION("axis_angle")
    {
      auto rot = Rot::from_axis_angle(Vec::fill(0.0, 0.0, 1.0, DTO), Scalar(M_PI / 2.0, DTO));
      auto v1 = Vec::fill(0.0, 1.0, 0.0, DTO);
      auto rv = v1.rotate(rot);
      auto v2 = Vec::fill(-1.0, 0.0, 0.0, DTO);
      REQUIRE(at::allclose(rv, v2));
    }

    SECTION("axis_angle_standard")
    {
      auto r1 = Rot::from_axis_angle(Vec::fill(2.0, -1.0, 1.0, DTO), Scalar(M_PI / 3.0, DTO));
      auto r2 =
          Rot::from_axis_angle_standard(Vec::fill(2.0, -1.0, 1.0, DTO), Scalar(M_PI / 3.0, DTO));
      REQUIRE(at::allclose(r1, r2));
    }

    SECTION("inverse")
    {
      auto a = Rot::fill(1.2, 3.1, -2.1, DTO);
      auto ab = a.batch_expand(B);
      auto b = Vec(at::rand(utils::add_shapes(B, 3), DTO));
      REQUIRE(at::allclose(b.rotate(a).rotate(a.inverse()), b));
      REQUIRE(at::allclose(b.rotate(ab).rotate(ab.inverse()), b.batch_expand(B)));
    }

    SECTION("euler_rodrigues")
    {
      auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto A = R2::fill(-0.91855865,
                        -0.1767767,
                        0.35355339,
                        0.30618622,
                        -0.88388348,
                        0.35355339,
                        0.25,
                        0.4330127,
                        0.8660254,
                        DTO);
      REQUIRE(at::allclose(a.euler_rodrigues(), A));
      REQUIRE(at::allclose(a.batch_expand(B).euler_rodrigues(), A.batch_expand(B)));
    }

    SECTION("deuler_rodrigues")
    {
      auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto apply = [](const Tensor & x) { return Rot(x).euler_rodrigues(); };
      auto dA_da = finite_differencing_derivative(apply, a);

      REQUIRE(at::allclose(a.deuler_rodrigues(), dA_da, 1.0e-4));
      REQUIRE(at::allclose(a.batch_expand(B).deuler_rodrigues(), dA_da.batch_expand(B), 1.0e-4));
    }

    SECTION("shadow")
    {
      auto a = Rot::fill(1.2, 3.1, -2.1, DTO);
      auto ab = a.batch_expand(B);
      auto b = Rot::fill(-0.07761966, -0.20051746, 0.13583441, DTO);

      SECTION("defintion")
      {
        REQUIRE(at::allclose(a.shadow(), b));
        REQUIRE(at::allclose(ab.shadow(), b));
      }
      SECTION("concept") { REQUIRE(at::allclose(a.euler_rodrigues(), b.euler_rodrigues())); }
      SECTION("derivative")
      {
        auto apply = [](const Tensor & x) { return Rot(x).shadow(); };
        auto dA = finite_differencing_derivative(apply, a);
        REQUIRE(at::allclose(a.dshadow(), dA, 1.0e-4));
      }
    }

    SECTION("rotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);
      auto vp = Rot::fill(1.48720771, -2.26086024, 1.02025338, DTO);

      auto rb = r.batch_expand(B);
      auto vb = v.batch_expand(B);
      auto vpb = vp.batch_expand(B);

      REQUIRE(at::allclose(v.rotate(r), vp));
      REQUIRE(at::allclose(vb.rotate(rb), vpb));
      REQUIRE(at::allclose(v.rotate(rb), vpb));
      REQUIRE(at::allclose(vb.rotate(r), vpb));
    }

    SECTION("drotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);

      auto rb = r.batch_expand(B);
      auto vb = v.batch_expand(B);

      auto apply = [v](const Tensor & x) { return v.rotate(Rot(x)); };
      auto dvp_dr = finite_differencing_derivative(apply, r);
      auto dvp_drb = dvp_dr.batch_expand(B);

      REQUIRE(at::allclose(v.drotate(r), dvp_dr, 1e-4));
      REQUIRE(at::allclose(vb.drotate(rb), dvp_drb, 1e-4));
      REQUIRE(at::allclose(v.drotate(rb), dvp_drb, 1e-4));
      REQUIRE(at::allclose(vb.drotate(r), dvp_drb, 1e-4));
    }

    SECTION("drotate_self")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);

      auto apply = [r](const Rot & x) { return x.rotate(r); };
      auto dvp_dr = finite_differencing_derivative(apply, v);

      REQUIRE(at::allclose(v.drotate_self(r), dvp_dr, 1e-4));
    }
  }

  SECTION("operator*")
  {
    auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
    auto b = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);
    auto c = Rot::fill(1.48720771, -2.26086024, 1.02025338, DTO);
    REQUIRE(at::allclose(a * b, c));
    REQUIRE(at::allclose(a.batch_expand(B) * b, c.batch_expand(B)));
    REQUIRE(at::allclose(a * b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) * b.batch_expand(B), c.batch_expand(B)));
  }
}
