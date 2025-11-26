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

#include "utils.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("WR2", "[tensors]")
{
  at::manual_seed(42);

  SECTION("constructors")
  {
    SECTION("R2")
    {
      auto S = R2::fill(0.0, 0.1, 3.4, -0.1, 0.0, -1.2, -3.4, 1.2, 0.0);
      auto s = WR2::fill(1.2, 3.4, -0.1);
      REQUIRE_THAT(WR2(S), test::allclose(s));
    }
  }

  SECTION("rotate")
  {
    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto w = WR2::fill(-0.2, 0.012, 0.15);
    auto W = R2(w);

    REQUIRE(at::allclose(w.rotate(r), WR2(W.rotate(r))));

    // rotate by MRP
    auto apply_r = [w](const Tensor & x) { return w.rotate(Rot(x)); };
    auto dwp_dr = finite_differencing_derivative(apply_r, r);
    REQUIRE(at::allclose(w.drotate(r), dwp_dr, 1e-4));

    // rotate by rotation matrix
    auto R = r.euler_rodrigues();
    auto apply_R = [w](const Tensor & x) { return w.rotate(R2(x)); };
    auto dwp_dR = finite_differencing_derivative(apply_R, R);
    REQUIRE(at::allclose(w.drotate(R), dwp_dR));
  }

  SECTION("exp_map")
  {
    auto w = WR2::fill(-0.2, 0.012, 0.15);
    auto expw = R2::fill(0.98873698,
                         -0.14963255,
                         -0.00304675,
                         0.14724505,
                         0.9689128,
                         0.19881371,
                         -0.02679696,
                         -0.19702309,
                         0.98003256);

    REQUIRE_THAT(w.exp_map().euler_rodrigues(), test::allclose(expw, 0, 1e-4));

    // zero maps to zero
    auto w0 = WR2::fill(0.0, 0.0, 0.0);
    REQUIRE_THAT(w0.exp_map(), test::allclose(Rot::identity(), 0, 1e-4));

    // dexp_map
    auto apply = [](const WR2 & x) { return x.exp_map(); };

    auto dexpw_dw = finite_differencing_derivative(apply, w);
    REQUIRE_THAT(w.dexp_map(), test::allclose(dexpw_dw, 0, 1e-4));

    auto dexpw0_dw0 = finite_differencing_derivative(apply, w0);
    REQUIRE_THAT(w0.dexp_map(), test::allclose(dexpw0_dw0, 0, 1e-4));

    auto ws = WR2::fill(1.0e-12, 0.0, 0.0);
    auto dexpws_dws = finite_differencing_derivative(apply, ws);
    REQUIRE_THAT(ws.dexp_map(), test::allclose(dexpws_dws, 0, 1e-4));
  }

  SECTION("operator()")
  {
    auto a = WR2::rand({}, {});
    auto b = R2(a);
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        REQUIRE_THAT(a(i, j), test::allclose(b(i, j)));
  }
}
