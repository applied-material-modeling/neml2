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

#include "neml2/tensors/MillerIndex.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("MillerIndex", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();
  const auto & IDTO = default_integer_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  auto m = MillerIndex::fill(1, 2, -1, IDTO);
  auto mb = m.batch_expand(B);

  SECTION("Convert to vector")
  {
    auto right = Vec::fill(1.0, 2.0, -1.0, DTO);
    REQUIRE(at::allclose(m.to_vec(), right));
    REQUIRE(at::allclose(mb.to_vec(), right.batch_expand(B)));
  }

  SECTION("Convert to normalized vector")
  {
    auto right = Vec::fill(1.0, 2.0, -1.0, DTO);
    right /= right.norm();
    REQUIRE(at::allclose(m.to_normalized_vec(), right));
    REQUIRE(at::allclose(mb.to_normalized_vec(), right.batch_expand(B)));
  }

  SECTION("GCD reduction")
  {
    auto a = MillerIndex::fill(2, 2, 6, IDTO);
    REQUIRE(at::all(a.reduce() == MillerIndex::fill(1, 1, 3, IDTO)).item<bool>());
    auto b = MillerIndex::fill(-2, 2, 6, IDTO);
    REQUIRE(at::all(b.reduce() == MillerIndex::fill(-1, 1, 3)).item<bool>());
    auto c = MillerIndex::fill(8, 4, 2, IDTO);
    REQUIRE(at::all(c.reduce() == MillerIndex::fill(4, 2, 1, IDTO)).item<bool>());
  }
}
