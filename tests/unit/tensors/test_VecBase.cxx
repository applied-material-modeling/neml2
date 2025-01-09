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

TEST_CASE("VecBase", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class VecBase")
  {
    SECTION("fill")
    {
      auto v1 = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto v2 = Vec::fill(Scalar(1.1, DTO), Scalar(2.2, DTO), Scalar(3.3, DTO));
      auto v3 = Vec::fill(
          Scalar::full(B, 1.1, DTO), Scalar::full(B, 2.2, DTO), Scalar::full(B, 3.3, DTO));
      REQUIRE(at::allclose(v1, v2));
      REQUIRE(at::allclose(v1.batch_expand(B), v3));
      REQUIRE(at::allclose(v2.batch_expand(B), v3));
    }

    SECTION("identity_map")
    {
      auto I = Vec::identity_map(DTO);
      auto a = Vec(at::rand(utils::add_shapes(B, 3), DTO));

      auto apply = [](const Tensor & x) { return x; };
      auto da_da = finite_differencing_derivative(apply, a);

      REQUIRE(at::allclose(I, da_da));
    }

    SECTION("operator()")
    {
      auto a = Vec(at::rand(utils::add_shapes(B, 3), DTO));
      REQUIRE(at::allclose(a(0), a.index({indexing::Ellipsis, 0})));
      REQUIRE(at::allclose(a(1), a.index({indexing::Ellipsis, 1})));
      REQUIRE(at::allclose(a(2), a.index({indexing::Ellipsis, 2})));
    }

    SECTION("dot")
    {
      auto a = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto b = Vec::fill(5.1, -3.3, 0.2, DTO);
      auto c = Scalar(-0.99, DTO);
      REQUIRE(at::allclose(a.dot(b), c));
      REQUIRE(at::allclose(a.batch_expand(B).dot(b), c.batch_expand(B)));
      REQUIRE(at::allclose(a.dot(b.batch_expand(B)), c.batch_expand(B)));
      REQUIRE(at::allclose(a.batch_expand(B).dot(b.batch_expand(B)), c.batch_expand(B)));
    }

    SECTION("cross")
    {
      auto a = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto b = Vec::fill(5.1, -3.3, 0.2, DTO);
      auto c = Vec::fill(11.33, 16.61, -14.85, DTO);
      REQUIRE(at::allclose(a.cross(b), c));
      REQUIRE(at::allclose(a.batch_expand(B).cross(b), c.batch_expand(B)));
      REQUIRE(at::allclose(a.cross(b.batch_expand(B)), c.batch_expand(B)));
      REQUIRE(at::allclose(a.batch_expand(B).cross(b.batch_expand(B)), c.batch_expand(B)));
    }

    SECTION("outer")
    {
      auto a = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto b = Vec::fill(5.1, -3.3, 0.2, DTO);
      auto c = R2::create(
          {{5.6100, -3.6300, 0.2200}, {11.2200, -7.2600, 0.4400}, {16.8300, -10.8900, 0.6600}},
          DTO);
      REQUIRE(at::allclose(a.outer(b), c));
      REQUIRE(at::allclose(a.batch_expand(B).outer(b), c.batch_expand(B)));
      REQUIRE(at::allclose(a.outer(b.batch_expand(B)), c.batch_expand(B)));
      REQUIRE(at::allclose(a.batch_expand(B).outer(b.batch_expand(B)), c.batch_expand(B)));
    }

    SECTION("norm_sq")
    {
      auto a = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto b = Scalar(16.94, DTO);
      REQUIRE(at::allclose(a.norm_sq(), b));
      REQUIRE(at::allclose(a.batch_expand(B).norm_sq(), b.batch_expand(B)));
    }

    SECTION("norm")
    {
      auto a = Vec::fill(1.1, 2.2, 3.3, DTO);
      auto b = Scalar(4.115823125451335, DTO);
      REQUIRE(at::allclose(a.norm(), b));
      REQUIRE(at::allclose(a.batch_expand(B).norm(), b.batch_expand(B)));
    }
  }
}
