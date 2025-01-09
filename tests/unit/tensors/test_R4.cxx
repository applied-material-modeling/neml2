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

TEST_CASE("R4", "[R4]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class R4")
  {
    SECTION("R4")
    {
      auto u = R4(at::rand(utils::add_shapes(B, 3, 3, 3, 3), DTO));
      // Symmetrize it
      auto s = (u + u.transpose_minor() + u.transpose(0, 1) + u.transpose(2, 3)) / 4.0;
      // Converting to SSR4 and then back should be equivalent to symmetrization
      REQUIRE(at::allclose(s, R4(SSR4(u))));
    }

    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto T = R4::create({{{{0.66112296, 0.67364277, 0.52908828},
                           {0.56724338, 0.58715151, 0.11093917},
                           {0.21574421, 0.15568454, 0.81052343}},
                          {{0.57389508, 0.46795234, 0.62969397},
                           {0.58735001, 0.96843709, 0.1604007},
                           {0.88311546, 0.0441955, 0.48777658}},
                          {{0.99507367, 0.56344149, 0.34286399},
                           {0.15020997, 0.15300364, 0.84086095},
                           {0.5106674, 0.45230156, 0.21724192}}},
                         {{{0.54456104, 0.18254561, 0.49353823},
                           {0.59161612, 0.81852437, 0.46011312},
                           {0.8643376, 0.71817923, 0.99371746}},
                          {{0.48442184, 0.62605832, 0.73174494},
                           {0.90427983, 0.21560154, 0.85167291},
                           {0.60321318, 0.70176223, 0.72316361}},
                          {{0.03911803, 0.284356, 0.47101786},
                           {0.23046833, 0.43203527, 0.80362567},
                           {0.10884239, 0.26013328, 0.64722489}}},
                         {{{0.97510859, 0.1980099, 0.82347827},
                           {0.15653814, 0.05652895, 0.58470749},
                           {0.08975475, 0.5209197, 0.59695489}},
                          {{0.40475775, 0.58923968, 0.68776156},
                           {0.84788879, 0.34349879, 0.65479406},
                           {0.51828743, 0.85120858, 0.887165}},
                          {{0.63091418, 0.04140195, 0.40599633},
                           {0.66631594, 0.2543073, 0.63205863},
                           {0.76469959, 0.27718685, 0.77058401}}}},
                        DTO);
    auto Tp = R4::create({{{{0.23820857, 0.43305693, -0.20977483},
                            {0.62563634, 0.54712896, 0.1482663},
                            {-0.42577276, 0.0763476, 0.77115534}},
                           {{-0.11860723, 0.05294212, -0.1852346},
                            {0.1659533, 0.55463045, -0.02926287},
                            {-0.4664211, 0.20061751, 0.08689772}},
                           {{-0.57907037, -0.27356366, 0.86360503},
                            {0.287918, -0.16939878, 0.27825703},
                            {0.66426806, 0.12291877, -1.10089594}}},
                          {{{-0.05775137, -0.08507278, 0.2028092},
                            {0.03165453, 0.15485068, -0.11123546},
                            {-0.72807784, -0.06492156, 0.72400724}},
                           {{0.22689592, -0.12497932, -0.26253125},
                            {-0.0691542, -0.56413159, -0.00848574},
                            {-0.09947468, 0.10061001, 0.12538374}},
                           {{0.32656294, -0.09888548, 0.087943},
                            {-0.20833318, 0.06218009, 0.27494329},
                            {-0.2485973, 0.12094771, -0.62021714}}},
                          {{{-0.61516067, 0.29228151, 0.84331687},
                            {-0.06538272, -0.08037612, -0.17996251},
                            {0.41248725, 0.19490796, -1.66034649}},
                           {{0.07809489, -0.24446264, 0.39108274},
                            {-0.45171636, 0.27742552, 0.03804866},
                            {0.50723862, 0.23988241, -0.89988478}},
                           {{0.69355161, -0.20550391, -1.19532462},
                            {0.15709077, -0.14514052, -0.46242684},
                            {-1.20970014, 0.18995295, 3.24473836}}}},
                         DTO);

    auto rb = r.batch_expand(B);
    auto Tb = T.batch_expand(B);
    auto Tpb = Tp.batch_expand(B);

    SECTION("rotate")
    {
      REQUIRE(at::allclose(T.rotate(r), Tp));
      REQUIRE(at::allclose(Tb.rotate(rb), Tpb));
      REQUIRE(at::allclose(T.rotate(rb), Tpb));
      REQUIRE(at::allclose(Tb.rotate(r), Tpb));
    }

    SECTION("drotate")
    {
      auto apply = [T](const Tensor & x) { return T.rotate(Rot(x)); };
      auto dTp_dr = finite_differencing_derivative(apply, r);
      auto dTp_drb = dTp_dr.batch_expand(B);

      REQUIRE(at::allclose(T.drotate(r), dTp_dr, /*rtol=*/0, /*atol=*/1e-4));
      REQUIRE(at::allclose(Tb.drotate(rb), dTp_drb, /*rtol=*/0, /*atol=*/1e-4));
      REQUIRE(at::allclose(T.drotate(rb), dTp_drb, /*rtol=*/0, /*atol=*/1e-4));
      REQUIRE(at::allclose(Tb.drotate(r), dTp_drb, /*rtol=*/0, /*atol=*/1e-4));
    }

    SECTION("drotate_self")
    {
      auto apply = [r](const Tensor & x) { return R4(x).rotate(r); };
      auto dT_dT = finite_differencing_derivative(apply, T);
      auto dT_dTb = dT_dT.batch_expand(B);

      REQUIRE(at::allclose(T.drotate_self(r), dT_dT, 1.0e-4, 1.0e-4));
      REQUIRE(at::allclose(Tb.drotate_self(r), dT_dTb, 1.0e-4, 1.0e-4));
      REQUIRE(at::allclose(T.drotate_self(rb), dT_dTb, /*rtol=*/1.0e-4, /*atol=*/1e-4));
      REQUIRE(at::allclose(Tb.drotate_self(r), dT_dTb, /*rtol=*/1.0e-4, /*atol=*/1e-4));
    }

    SECTION("operator()")
    {
      auto u = R4(at::rand(utils::add_shapes(B, 3, 3, 3, 3), DTO));
      auto s1 = (u + u.transpose_minor() + u.transpose(0, 1) + u.transpose(2, 3)) / 4.0;
      auto s2 = R4(SSR4(u));
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
            for (Size l = 0; l < 3; l++)
              REQUIRE(at::allclose(s1(i, j, k, l), s2(i, j, k, l)));
    }

    SECTION("transpose_minor")
    {
      auto u = R4(at::rand(utils::add_shapes(B, 3, 3, 3, 3), DTO));
      auto ut = u.transpose_minor();
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
            for (Size l = 0; l < 3; l++)
              REQUIRE(at::allclose(u(i, j, k, l), ut(j, i, l, k)));
    }

    SECTION("transpose_major")
    {
      auto u = R4(at::rand(utils::add_shapes(B, 3, 3, 3, 3), DTO));
      auto ut = u.transpose_major();
      for (Size i = 0; i < 3; i++)
        for (Size j = 0; j < 3; j++)
          for (Size k = 0; k < 3; k++)
            for (Size l = 0; l < 3; l++)
              REQUIRE(at::allclose(u(i, j, k, l), ut(k, l, i, j)));
    }
  }
}
