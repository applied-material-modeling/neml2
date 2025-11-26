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

#include "neml2/misc/defaults.h"
#include "neml2/tensors/functions/operators.h"
#include "utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/functions/norm.h"

using namespace neml2;

TEST_CASE("Rot", "[tensors]")
{
  SECTION("identity")
  {
    auto a = Rot::identity();
    auto b = Vec::rand({2, 3}, {1, 4});
    // Rotate by an identity rotation should be a no-op
    REQUIRE_THAT(b.rotate(a), test::allclose(b));
  }

  SECTION("rotation_from_to")
  {
    auto va = Vec::fill(1.0, 2.0, 3.0);
    va /= neml2::norm(va);
    auto vb = Vec::fill(4.0, 5.0, 6.0);
    vb /= neml2::norm(vb);
    auto r = Rot::rotation_from_to(va, vb);
    REQUIRE_THAT(va.rotate(r), test::allclose(vb));
  }

  SECTION("axis_angle")
  {
    auto rot =
        Rot::axis_angle(Vec::fill(0.0, 0.0, 1.0), Scalar(M_PI / 2.0, default_tensor_options()));
    auto v1 = Vec::fill(0.0, 1.0, 0.0);
    auto rv = v1.rotate(rot);
    auto v2 = Vec::fill(-1.0, 0.0, 0.0);
    REQUIRE_THAT(rv, test::allclose(v2));
  }

  SECTION("axis_angle_standard")
  {
    auto r1 =
        Rot::axis_angle(Vec::fill(2.0, -1.0, 1.0), Scalar(M_PI / 3.0, default_tensor_options()));
    auto r2 = Rot::axis_angle_standard(Vec::fill(2.0, -1.0, 1.0),
                                       Scalar(M_PI / 3.0, default_tensor_options()));
    REQUIRE_THAT(r1, test::allclose(r2));
  }

  SECTION("euler_rodrigues")
  {
    auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto A = R2::fill(-0.91855865,
                      -0.1767767,
                      0.35355339,
                      0.30618622,
                      -0.88388348,
                      0.35355339,
                      0.25,
                      0.4330127,
                      0.8660254);
    REQUIRE_THAT(a.euler_rodrigues(), test::allclose(A));

    auto apply = [](const Tensor & x) { return Rot(x).euler_rodrigues(); };
    auto dA_da = finite_differencing_derivative(apply, a);
    REQUIRE_THAT(a.deuler_rodrigues(), test::allclose(dA_da, 1.0e-4));
  }

  SECTION("shadow")
  {
    auto a = Rot::fill(1.2, 3.1, -2.1);
    auto b = Rot::fill(-0.07761966, -0.20051746, 0.13583441);

    REQUIRE_THAT(a.shadow(), test::allclose(b));
    REQUIRE_THAT(a.euler_rodrigues(), test::allclose(b.euler_rodrigues()));

    auto apply = [](const Tensor & x) { return Rot(x).shadow(); };
    auto dA = finite_differencing_derivative(apply, a);
    REQUIRE_THAT(a.dshadow(), test::allclose(dA, 1.0e-4));
  }

  SECTION("rotate")
  {
    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009);
    auto vp = Rot::fill(1.48720771, -2.26086024, 1.02025338);

    REQUIRE_THAT(v.rotate(r), test::allclose(vp));

    auto apply1 = [v](const Tensor & x) { return v.rotate(Rot(x)); };
    auto dvp_dr = finite_differencing_derivative(apply1, r);
    REQUIRE_THAT(v.drotate(r), test::allclose(dvp_dr, 1e-4));

    auto apply2 = [r](const Rot & x) { return x.rotate(r); };
    auto dvp_dr_self = finite_differencing_derivative(apply2, v);
    REQUIRE_THAT(v.drotate_self(r), test::allclose(dvp_dr_self, 1e-4));
  }

  SECTION("operator*")
  {
    auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto b = Rot::fill(-0.32366123, -0.15961206, 0.86937009);
    auto c = Rot::fill(1.48720771, -2.26086024, 1.02025338);
    REQUIRE_THAT(a * b, test::allclose(c));
  }
}
