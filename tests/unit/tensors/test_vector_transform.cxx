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

#include "neml2/tensors/Transformable.h"

#include "neml2/tensors/R2.h"
#include "neml2/tensors/Rot.h"
#include "neml2/tensors/Vec.h"

using namespace neml2;

TEST_CASE("Symmetry transforms", "[crystallography]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape

  SECTION("Vec")
  {
    auto v = Vec::fill(1.0, -2.0, 3.0, DTO);
    auto vb = v.batch_expand(B);
    SECTION("Identity")
    {
      auto op = identity_transform(DTO);
      REQUIRE(at::allclose(v.transform(op), v));
      REQUIRE(at::allclose(vb.transform(op), vb));
    }
    // 90 about z
    auto r = Rot::fill(0.0, 0, 0.41421356, DTO);
    SECTION("ProperRotation")
    {
      auto op = proper_rotation_transform(r);
      auto correct = Vec::fill(2.0, 1.0, 3.0, DTO);
      REQUIRE(at::allclose(v.transform(op), correct));
      REQUIRE(at::allclose(vb.transform(op), correct.batch_expand(B)));
    }
    // 90 about z
    SECTION("ImproperRotation")
    {
      auto op = improper_rotation_transform(r);
      auto correct = Vec::fill(2.0, 1.0, -3.0, DTO);
      REQUIRE(at::allclose(v.transform(op), correct));
      REQUIRE(at::allclose(vb.transform(op), correct.batch_expand(B)));
    }
    SECTION("Inversion")
    {
      auto op = inversion_transform(DTO);
      auto correct = Vec::fill(-1.0, 2.0, -3.0, DTO);
      REQUIRE(at::allclose(v.transform(op), correct));
      REQUIRE(at::allclose(vb.transform(op), correct.batch_expand(B)));
    }
  }
}
