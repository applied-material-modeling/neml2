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

#include "neml2/tensors/Derivative.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("Derivative", "[tensors]")
{
  SECTION("derivative_name")
  {
    REQUIRE(derivative_name<1>("y", {"x"}) == "d(y)/d(x)");
    REQUIRE(derivative_name<2>("y", {"x", "z"}) == "d2(y)/d(x)d(z)");
  }

  SECTION("name")
  {
    Derivative<1> deriv(2,
                        {2, 2},
                        {TensorShape{2, 2}, TensorShape{2, 2}},
                        {TensorShape{}, TensorShape{}},
                        "y",
                        {"x"});
    REQUIRE(deriv.name() == "d(y)/d(x)");
  }

  SECTION("var_name")
  {
    Derivative<1> deriv(2,
                        {2, 2},
                        {TensorShape{2, 2}, TensorShape{2, 2}},
                        {TensorShape{}, TensorShape{}},
                        "y",
                        {"x"});
    REQUIRE(deriv.var_name() == "y");
  }

  SECTION("arg_name")
  {
    Derivative<1> deriv(2,
                        {2, 2},
                        {TensorShape{2, 2}, TensorShape{2, 2}},
                        {TensorShape{}, TensorShape{}},
                        "y",
                        {"x"});
    REQUIRE(deriv.arg_name(0) == "x");
  }

  SECTION("defined")
  {
    Derivative<1> deriv(2,
                        {2, 2},
                        {TensorShape{2, 2}, TensorShape{2, 2}},
                        {TensorShape{}, TensorShape{}},
                        "y",
                        {"x"});
    REQUIRE(!deriv.defined());
    deriv = Tensor::create({{1.0, 2.0}, {3.0, 4.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
    REQUIRE(deriv.defined());
  }

  SECTION("clear")
  {
    Derivative<1> deriv(2,
                        {2, 2},
                        {TensorShape{2, 2}, TensorShape{2, 2}},
                        {TensorShape{}, TensorShape{}},
                        "y",
                        {"x"});
    deriv = Tensor::create({{1.0, 2.0}, {3.0, 4.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
    REQUIRE(deriv.defined());
    deriv.clear();
    REQUIRE(!deriv.defined());
  }

  SECTION("is_intrsc_intmd_broadcast")
  {
    SECTION("broadcast")
    {
      Derivative<1> deriv(2,
                          {2, 2},
                          {TensorShape{2, 2}, TensorShape{2, 2}},
                          {TensorShape{}, TensorShape{}},
                          "y",
                          {"x"});
      REQUIRE(deriv.is_intrsc_intmd_broadcast());
    }

    SECTION("full")
    {
      Derivative<1> deriv(4,
                          {2, 2},
                          {TensorShape{2, 2}, TensorShape{2, 2}},
                          {TensorShape{}, TensorShape{}},
                          "y",
                          {"x"});
      REQUIRE(!deriv.is_intrsc_intmd_broadcast());
    }
  }

  SECTION("reinterpret")
  {
    const auto var_intmd = TensorShape{2, 2, 2};
    const auto arg_intmd = TensorShape{2, 2, 2};
    Derivative<1> deriv(
        2, {2, 2}, {var_intmd, arg_intmd}, {TensorShape{}, TensorShape{}}, "y", {"x"});
    const auto t = Tensor::create({{1.0, 2.0}, {3.0, 4.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
    deriv = t;

    SECTION("no-op")
    {
      const auto r = deriv.reinterpret(0);
      REQUIRE(r.intrsc_intmd_dim() == deriv.intrsc_intmd_dim());
      REQUIRE(r.tensor().intmd_sizes() == deriv.tensor().intmd_sizes());
      REQUIRE_THAT(r.tensor(), test::allclose(deriv.tensor()));
    }

    SECTION("expand")
    {
      const auto r = deriv.reinterpret(1);
      REQUIRE(r.intrsc_intmd_dim() == 3);
      REQUIRE(r.var_intrsc_intmd_dim() == 3);
      REQUIRE(r.arg_intrsc_intmd_dim(0) == 3);
      REQUIRE(r.tensor().intmd_sizes() == TensorShapeRef{2, 2, 2});

      for (Size i = 0; i < 2; ++i)
        for (Size j = 0; j < 2; ++j)
          for (Size k = 0; k < 2; ++k)
            REQUIRE(r.tensor().intmd_index({i, j, k}).item<double>() ==
                    t.intmd_index({i, j}).item<double>());
    }
  }

  SECTION("fullify")
  {
    SECTION("no-op")
    {
      const auto t =
          Tensor::create({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
      const auto out = neml2::fullify<2>(t, /*iid=*/2, {TensorShape{2}, TensorShape{3}});
      REQUIRE(out.intmd_sizes() == t.intmd_sizes());
      REQUIRE_THAT(out, test::allclose(t));
    }

    SECTION("broadcast N==1")
    {
      const auto t = Tensor::create({{1.0, 2.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
      const auto out = neml2::fullify<1>(t, /*iid=*/2, {TensorShape{2, 2}});
      REQUIRE(out.intmd_sizes() == TensorShapeRef{2, 2});
      REQUIRE(out.intmd_index({0, 0}).item<double>() == 1.0);
      REQUIRE(out.intmd_index({1, 0}).item<double>() == 1.0);
      REQUIRE(out.intmd_index({0, 1}).item<double>() == 2.0);
      REQUIRE(out.intmd_index({1, 1}).item<double>() == 2.0);
    }

    SECTION("broadcast N>1")
    {
      Derivative<1> deriv(2,
                          {2, 2},
                          {TensorShape{2, 2}, TensorShape{2, 2}},
                          {TensorShape{}, TensorShape{}},
                          "y",
                          {"x"});
      const auto t = Tensor::create({{1.0, 2.0}, {3.0, 4.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/2);
      deriv = t;

      const auto out = deriv.fullify();
      REQUIRE(out.intmd_sizes() == TensorShapeRef{2, 2, 2, 2});

      for (Size i = 0; i < 2; ++i)
        for (Size j = 0; j < 2; ++j)
          for (Size k = 0; k < 2; ++k)
            for (Size l = 0; l < 2; ++l)
            {
              const auto expected = (i == k && j == l) ? t.intmd_index({i, j}).item<double>() : 0.0;
              REQUIRE(out.intmd_index({i, j, k, l}).item<double>() == expected);
            }
    }
  }
}
