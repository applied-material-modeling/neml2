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

#include <cmath>

#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("SparseTensorList", "[equation_systems]")
{
  SECTION("options")
  {
    SparseTensorList empty(2);
    REQUIRE_THROWS(empty.options());

    SparseTensorList list(2);
    list[1] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    const auto opts = list.options();
    REQUIRE(opts.dtype() == list[1].options().dtype());
    REQUIRE(opts.device() == list[1].options().device());
  }

  SECTION("data")
  {
    SparseTensorList list(2);
    list[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto data = list.data();
    REQUIRE(data.size() == list.size());
    REQUIRE(data[0].defined());
    REQUIRE_THAT(data[0], test::allclose(list[0]));
    REQUIRE(!data[1].defined());
  }

  SECTION("unary minus")
  {
    SparseTensorList list(2);
    list[0] = Tensor::create({1.0, -2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto neg = -list;
    REQUIRE(neg[0].defined());
    REQUIRE_THAT(neg[0], test::allclose(Tensor::create({-1.0, 2.0}, 0, 0)));
    REQUIRE(!neg[1].defined());
  }

  SECTION("addition")
  {
    SparseTensorList a(2);
    SparseTensorList b(2);
    a[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b[0] = Tensor::create({3.0, 4.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto c = a + b;
    REQUIRE(c[0].defined());
    REQUIRE_THAT(c[0], test::allclose(Tensor::create({4.0, 6.0}, 0, 0)));
    REQUIRE(c[1].defined());
    REQUIRE_THAT(c[1], test::allclose(b[1]));

    SparseTensorList bad(1);
    REQUIRE_THROWS(a + bad);
  }

  SECTION("scalar multiplication")
  {
    SparseTensorList list(2);
    list[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto s = Scalar(2.0, list[0].options());
    const auto left = s * list;
    const auto right = list * s;
    REQUIRE_THAT(left[0], test::allclose(Tensor::create({2.0, 4.0}, 0, 0)));
    REQUIRE_THAT(right[0], test::allclose(Tensor::create({2.0, 4.0}, 0, 0)));
    REQUIRE(!left[1].defined());
    REQUIRE(!right[1].defined());
  }

  SECTION("inner/norm")
  {
    SparseTensorList a(2);
    SparseTensorList b(2);
    a[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b[0] = Tensor::create({3.0, 4.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    a[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto s = inner(a, b);
    REQUIRE(s.item<double>() == 11.0);

    const auto ns = norm_sq(a);
    REQUIRE(ns.item<double>() == 30.0);

    const auto n = norm(a);
    REQUIRE(n.item<double>() == std::sqrt(30.0));
  }
}
