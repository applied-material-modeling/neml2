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

#include "neml2/misc/defaults.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/tensors.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("SSR4", "[tensors]")
{
  at::manual_seed(42);

  SECTION("constructors")
  {
    SECTION("R4")
    {
      auto u = R4::rand({2, 3}, {1, 2});
      // Symmetrize it
      auto s = (u + u.transpose_minor() + u.transpose(0, 1) + u.transpose(2, 3)) / 4.0;
      // Converting to SSR4 should be equivalent to symmetrization
      REQUIRE_THAT(R4(SSR4(u)), test::allclose(s));
    }
  }

  SECTION("identity")
  {
    auto a = SSR4::identity();
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
          for (Size l = 0; l < 3; l++)
            REQUIRE_THAT(
                a(i, j, k, l),
                test::allclose(Scalar(i == j && k == l ? 1.0 : 0.0, default_tensor_options())));
  }

  SECTION("identity_sym")
  {
    auto a = SSR4::identity_sym();
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
          for (Size l = 0; l < 3; l++)
            REQUIRE_THAT(a(i, j, k, l),
                         test::allclose(
                             Scalar((i == k && j == l ? 0.5 : 0.0) + (i == l && j == k ? 0.5 : 0.0),
                                    default_tensor_options())));
  }

  SECTION("identity_vol")
  {
    auto a = SSR4::identity_vol();
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
          for (Size l = 0; l < 3; l++)
            REQUIRE_THAT(
                a(i, j, k, l),
                test::allclose(Scalar(i == j && k == l ? 1.0 / 3 : 0.0, default_tensor_options())));
  }

  SECTION("identity_dev")
  {
    auto a = SSR4::identity_dev();
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
          for (Size l = 0; l < 3; l++)
            REQUIRE_THAT(a(i, j, k, l),
                         test::allclose(Scalar((i == k && j == l ? 0.5 : 0.0) +
                                                   (i == l && j == k ? 0.5 : 0.0) -
                                                   (i == j && k == l ? 1.0 / 3 : 0.0),
                                               default_tensor_options())));
  }

  SECTION("isotropic_E_nu")
  {
    Scalar E(100, default_tensor_options());
    Scalar nu(0.3, default_tensor_options());
    auto correct = SSR4::create({{134.6154, 57.6923, 57.6923, 0.0000, 0.0000, 0.0000},
                                 {57.6923, 134.6154, 57.6923, 0.0000, 0.0000, 0.0000},
                                 {57.6923, 57.6923, 134.6154, 0.0000, 0.0000, 0.0000},
                                 {0.0000, 0.0000, 0.0000, 76.9231, 0.0000, 0.0000},
                                 {0.0000, 0.0000, 0.0000, 0.0000, 76.9231, 0.0000},
                                 {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 76.9231}},
                                0);
    REQUIRE_THAT(SSR4::isotropic_E_nu(E, nu), test::allclose(correct));
  }

  SECTION("rotate")
  {

    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto T =
        SSR4::create({{0.66086749, 0.26509302, 0.55764353, 0.27368709, 0.16527339, 0.18229984},
                      {0.2164092, 0.7357522, 0.29142165, 0.64753131, 0.96644071, 0.7476113},
                      {0.49247497, 0.8989371, 0.56977659, 0.45106647, 0.07075565, 0.20201865},
                      {0.83117451, 0.4132504, 0.92118474, 0.81776138, 0.16917866, 0.85560904},
                      {0.63618107, 0.80588965, 0.53258787, 0.45440311, 0.7853135, 0.1011699},
                      {0.78730947, 0.38979234, 0.61653301, 0.98293833, 0.90139196, 0.08489829}},
                     0);
    auto Tp = SSR4::create({{4.15364825e-01,
                             -2.83211571e-02,
                             4.62312959e-01,
                             2.38964778e-02,
                             -4.02181770e-02,
                             -1.22499802e-01},
                            {-4.05881972e-01,
                             3.61349504e-01,
                             5.90686850e-02,
                             -1.95566720e-01,
                             -4.96611003e-01,
                             2.86345828e-01},
                            {9.69341009e-01,
                             6.01339282e-01,
                             2.25380261e+00,
                             -6.09222202e-01,
                             -8.49813214e-01,
                             4.47490904e-02},
                            {-1.50202034e-01,
                             4.50361133e-01,
                             -7.49992554e-01,
                             1.22987450e-01,
                             5.56251500e-01,
                             -2.83398279e-01},
                            {-2.64069598e-01,
                             -3.44878006e-01,
                             -1.34724573e+00,
                             -2.13162040e-03,
                             8.93062880e-01,
                             4.85206282e-01},
                            {-3.39076103e-01,
                             8.79393192e-02,
                             4.02061883e-01,
                             -1.67154634e-01,
                             -4.12658966e-01,
                             -3.92197835e-01}},
                           0);

    REQUIRE(at::allclose(T.rotate(r), Tp));

    auto apply1 = [T](const Tensor & x) { return T.rotate(Rot(x)); };
    auto dTp_dr = finite_differencing_derivative(apply1, r);
    REQUIRE_THAT(T.drotate(r), test::allclose(dTp_dr, /*rtol=*/0, /*atol=*/1e-4));

    auto apply2 = [r](const Tensor & x) { return SSR4(x).rotate(r); };
    auto dT_dT = finite_differencing_derivative(apply2, T);
    REQUIRE_THAT(T.drotate_self(r), test::allclose(dT_dT, /*rtol=*/0, /*atol=*/1.0e-4));
  }

  SECTION("operator()")
  {
    auto a = SSR4::rand({}, {});
    auto b = R4(a);
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
          for (Size l = 0; l < 3; l++)
            REQUIRE_THAT(a(i, j, k, l), test::allclose(b(i, j, k, l)));
  }

  SECTION("transpose_minor")
  {
    // no-op
  }

  SECTION("transpose_major")
  {
    auto a = SSR4::rand({}, {});
    auto b = R4(a).transpose(0, 2).transpose(1, 3);
    REQUIRE_THAT(R4(a.transpose_major()), test::allclose(b));
  }
}
