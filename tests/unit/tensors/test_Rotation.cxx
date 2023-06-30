// Copyright 2023, UChicago Argonne, LLC
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

#include <catch2/catch.hpp>

#include "neml2/tensors/Rotation.h"
#include "neml2/tensors/Scalar.h"

using namespace neml2;

TEST_CASE("Rotation", "[Rotation]")
{
  SECTION("inverse rotations are in fact inverses")
  {
    SECTION("identity is zero")
    {
      Rotation a = Rotation::identity();
      REQUIRE(torch::allclose(a, torch::zeros_like(a)));
    }

    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2), Scalar(3.1), Scalar(-2.1));
      Rotation b = a.inverse();

      REQUIRE(torch::allclose(a * b, Rotation::identity()));
    }
    SECTION("batched")
    {
      Rotation a = Rotation::init(Scalar(torch::tensor({{1.2}, {-0.5}}, TorchDefaults)),
                                  Scalar(torch::tensor({{3.1}, {-1.6}}, TorchDefaults)),
                                  Scalar(torch::tensor({{-2.1}, {0.5}}, TorchDefaults)));
      Rotation b = a.inverse();

      REQUIRE(torch::allclose(a * b, Rotation::identity()));
    }
  }
  SECTION("test composition of rotations")
  {
    SECTION("unbatched, operator")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      Rotation b = Rotation::init(Scalar(-5.68010824), Scalar(-2.8011194), Scalar(15.25705169));
      Rotation c = Rotation::init(Scalar(-0.40390244), Scalar(0.61401441), Scalar(-0.27708492));

      REQUIRE(torch::allclose(a * b, c));
    }
    SECTION("batched, operator")
    {
      Rotation a =
          Rotation::init(Scalar(torch::tensor({{1.2496889}, {-2.74440729}}, TorchDefaults)),
                         Scalar(torch::tensor({{1.62862628}, {-1.10086082}}, TorchDefaults)),
                         Scalar(torch::tensor({{7.59575411}, {-14.83201462}}, TorchDefaults)));
      Rotation b =
          Rotation::init(Scalar(torch::tensor({{-5.68010824}, {0.97525904}}, TorchDefaults)),
                         Scalar(torch::tensor({{-2.8011194}, {0.05227498}}, TorchDefaults)),
                         Scalar(torch::tensor({{15.25705169}, {-2.83462851}}, TorchDefaults)));
      Rotation c =
          Rotation::init(Scalar(torch::tensor({{-0.40390244}, {-0.05551478}}, TorchDefaults)),
                         Scalar(torch::tensor({{0.61401441}, {0.60802679}}, TorchDefaults)),
                         Scalar(torch::tensor({{-0.27708492}, {0.43687898}}, TorchDefaults)));

      REQUIRE(torch::allclose(a * b, c));
    }

    SECTION("unbatched, apply")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      Rotation b = Rotation::init(Scalar(-5.68010824), Scalar(-2.8011194), Scalar(15.25705169));
      Rotation c = Rotation::init(Scalar(-0.40390244), Scalar(0.61401441), Scalar(-0.27708492));

      REQUIRE(torch::allclose(a.apply(b), c));
    }
    SECTION("batched, apply")
    {
      Rotation a =
          Rotation::init(Scalar(torch::tensor({{1.2496889}, {-2.74440729}}, TorchDefaults)),
                         Scalar(torch::tensor({{1.62862628}, {-1.10086082}}, TorchDefaults)),
                         Scalar(torch::tensor({{7.59575411}, {-14.83201462}}, TorchDefaults)));
      Rotation b =
          Rotation::init(Scalar(torch::tensor({{-5.68010824}, {0.97525904}}, TorchDefaults)),
                         Scalar(torch::tensor({{-2.8011194}, {0.05227498}}, TorchDefaults)),
                         Scalar(torch::tensor({{15.25705169}, {-2.83462851}}, TorchDefaults)));
      Rotation c =
          Rotation::init(Scalar(torch::tensor({{-0.40390244}, {-0.05551478}}, TorchDefaults)),
                         Scalar(torch::tensor({{0.61401441}, {0.60802679}}, TorchDefaults)),
                         Scalar(torch::tensor({{-0.27708492}, {0.43687898}}, TorchDefaults)));

      REQUIRE(torch::allclose(a.apply(b), c));
    }
  }

  SECTION("test conversion to matrix")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      auto Ap = R2(torch::tensor({{{-0.91855865, -0.1767767, 0.35355339},
                                   {0.30618622, -0.88388348, 0.35355339},
                                   {0.25, 0.4330127, 0.8660254}}},
                                 TorchDefaults));
      REQUIRE(torch::allclose(a.to_R2(), Ap));
    }
  }

  SECTION("rotate vectors")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      Vector v = Vector(torch::tensor({{1.0, -2.0, 3.0}}, TorchDefaults));

      Vector vp = Vector(torch::tensor({{0.495655, 3.13461, 1.98205}}, TorchDefaults));

      REQUIRE(torch::allclose(a.apply(v), vp));
    }
  }

  SECTION("rotate R2s")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411)); 
      R2 T = R2(torch::tensor({{
                              {1.0,2.0,3.0},
                              {4.0,5.0,6.0},
                              {7.0,8.0,9.0}}}, TorchDefaults));
      R2 U = R2(torch::tensor({{
                              {-1.02332, -0.0592151, -0.290549},
                              {0.440785, 0.208734, -1.65399},
                              {-5.14556, -2.0769, 15.8146}}}, TorchDefaults));
      REQUIRE(torch::allclose(a.apply(T), U));
    }
  }

  SECTION("rotate SymR2s")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411)); 
      SymR2 T = SymR2::init(R2(torch::tensor({{
                              {1.0,2.0,3.0},
                              {4.0,5.0,6.0},
                              {7.0,8.0,9.0}}}, TorchDefaults)));
      SymR2 U = SymR2::init(-1.02332, 0.208734, 15.8146, -1.86545, -2.71806, 0.190785);
      REQUIRE(torch::allclose(a.apply(T), U));
    }
  }

  SECTION("rotate R4s")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      R4 T = R4(torch::tensor({{{{{0.66112296, 0.67364277, 0.52908828},
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
                                  {0.76469959, 0.27718685, 0.77058401}}}}},
                              TorchDefaults));
      R4 U = R4(torch::tensor({{{{{0.23820857, 0.43305693, -0.20977483},
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
                                  {-1.20970014, 0.18995295, 3.24473836}}}}},
                              TorchDefaults));

      REQUIRE(torch::allclose(a.apply(T), U));
    }
  }

  SECTION("rotate SymSymR4s")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      SymSymR4 T = SymSymR4(torch::tensor(
          {{{0.66086749, 0.26509302, 0.55764353, 0.27368709, 0.16527339, 0.18229984},
            {0.2164092, 0.7357522, 0.29142165, 0.64753131, 0.96644071, 0.7476113},
            {0.49247497, 0.8989371, 0.56977659, 0.45106647, 0.07075565, 0.20201865},
            {0.83117451, 0.4132504, 0.92118474, 0.81776138, 0.16917866, 0.85560904},
            {0.63618107, 0.80588965, 0.53258787, 0.45440311, 0.7853135, 0.1011699},
            {0.78730947, 0.38979234, 0.61653301, 0.98293833, 0.90139196, 0.08489829}}},
          TorchDefaults));
      SymSymR4 U = SymSymR4(torch::tensor({{{4.15364825e-01,
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
                                             -3.92197835e-01}}},
                                          TorchDefaults));

      REQUIRE(torch::allclose(a.apply(T), U));
    }
  }
}
