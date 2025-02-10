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
#include "neml2/tensors/functions/pow.h"

using namespace neml2;

TEST_CASE("Scalar", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class Scalar")
  {
    SECTION("Scalar")
    {
      Scalar a(3.3, DTO);
      auto b = Scalar::full(3.3, DTO);
      REQUIRE(at::allclose(a, b));
    }

    SECTION("identity_map")
    {
      auto I = Scalar::identity_map(DTO);
      auto a = Scalar(at::rand(B, DTO));

      auto apply = [](const Tensor & x) { return x; };
      auto da_da = finite_differencing_derivative(apply, a);

      REQUIRE(at::allclose(I, da_da));
    }
  }

  SECTION("operator+")
  {
    Scalar a(3.3, DTO);
    auto b = Tensor::full({3, 2, 1}, -5.2, DTO);
    auto c = Tensor::full({3, 2, 1}, -1.9, DTO);

    REQUIRE(at::allclose(a + b, c));
    REQUIRE(at::allclose(a.batch_expand(B) + b, c.batch_expand(B)));
    REQUIRE(at::allclose(a + b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) + b.batch_expand(B), c.batch_expand(B)));

    REQUIRE(at::allclose(b + a, c));
    REQUIRE(at::allclose(b.batch_expand(B) + a, c.batch_expand(B)));
    REQUIRE(at::allclose(b + a.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(b.batch_expand(B) + a.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator-")
  {
    Scalar a(3.3, DTO);
    auto b = Tensor::full({3, 2, 1}, -5.2, DTO);
    auto c = Tensor::full({3, 2, 1}, 8.5, DTO);

    REQUIRE(at::allclose(a - b, c));
    REQUIRE(at::allclose(a.batch_expand(B) - b, c.batch_expand(B)));
    REQUIRE(at::allclose(a - b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) - b.batch_expand(B), c.batch_expand(B)));

    REQUIRE(at::allclose(b - a, -c));
    REQUIRE(at::allclose(b.batch_expand(B) - a, -c.batch_expand(B)));
    REQUIRE(at::allclose(b - a.batch_expand(B), -c.batch_expand(B)));
    REQUIRE(at::allclose(b.batch_expand(B) - a.batch_expand(B), -c.batch_expand(B)));
  }

  SECTION("operator*")
  {
    Scalar a(3.3, DTO);
    auto b = Tensor::full({3, 2, 1}, -5.2, DTO);
    auto c = Tensor::full({3, 2, 1}, -17.16, DTO);

    REQUIRE(at::allclose(a * b, c));
    REQUIRE(at::allclose(a.batch_expand(B) * b, c.batch_expand(B)));
    REQUIRE(at::allclose(a * b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) * b.batch_expand(B), c.batch_expand(B)));

    REQUIRE(at::allclose(b * a, c));
    REQUIRE(at::allclose(b.batch_expand(B) * a, c.batch_expand(B)));
    REQUIRE(at::allclose(b * a.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(b.batch_expand(B) * a.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator/")
  {
    Scalar a(3.3, DTO);
    auto b = Tensor::full({3, 2, 1}, -5.2, DTO);
    auto c = Tensor::full({3, 2, 1}, -0.6346153846153846, DTO);

    REQUIRE(at::allclose(a / b, c));
    REQUIRE(at::allclose(a.batch_expand(B) / b, c.batch_expand(B)));
    REQUIRE(at::allclose(a / b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) / b.batch_expand(B), c.batch_expand(B)));

    REQUIRE(at::allclose(b / a, 1.0 / c));
    REQUIRE(at::allclose(b.batch_expand(B) / a, 1.0 / c.batch_expand(B)));
    REQUIRE(at::allclose(b / a.batch_expand(B), 1.0 / c.batch_expand(B)));
    REQUIRE(at::allclose(b.batch_expand(B) / a.batch_expand(B), 1.0 / c.batch_expand(B)));
  }

  SECTION("pow")
  {
    Scalar a(3.3, DTO);
    auto b = Tensor::full({2, 2}, 2.0, DTO);
    auto c = Tensor::full({2, 2}, 9.849155306759329, DTO);
    REQUIRE(at::allclose(neml2::pow(b, a), c));
    REQUIRE(at::allclose(neml2::pow(b.batch_expand(B), a), c.batch_expand(B)));
    REQUIRE(at::allclose(neml2::pow(b, a.batch_expand(B)), c.batch_expand(B)));
    REQUIRE(at::allclose(neml2::pow(b.batch_expand(B), a.batch_expand(B)), c.batch_expand(B)));
  }
}
