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

#include "neml2/tensors/shape_utils.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/Tensor.h"

using namespace neml2;

TEST_CASE("shape_utils", "[tensors]")
{
  const auto & DTO = default_tensor_options();

  SECTION("broadcast_batch_dim")
  {
    TensorShape a = {};
    TensorShape b = {1, 2};
    TensorShape c = {3};
    TensorShape d = {4, 5, 6};

    // Create some tensors with the above batch shapes, the base shapes should not matter.
    auto A = Tensor::empty(a, {5, 3}, DTO);
    auto B = Tensor::empty(b, {1, 2}, DTO);
    auto C = Tensor::empty(c, {}, DTO);
    auto D = Tensor::empty(d, {3, 5, 6}, DTO);

    REQUIRE(utils::broadcast_batch_dim(A) == 0);
    REQUIRE(utils::broadcast_batch_dim(A, B) == 2);
    REQUIRE(utils::broadcast_batch_dim(A, B, C) == 2);
    REQUIRE(utils::broadcast_batch_dim(A, B, C, D) == 3);
  }

  SECTION("broadcastable")
  {
    // Broadcastable
    // 1. Same base shapes
    // 2. Batch shapes are broadcastable
    auto A1 = Tensor::empty({1, 2}, {5, 3}, DTO);
    auto B1 = Tensor::empty({1}, {5, 3}, DTO);
    REQUIRE(utils::broadcastable(A1, B1));

    // Broadcastable
    // 1. Same base shapes
    // 2. Batch shapes (can be empty) are broadcastable
    auto A2 = Tensor::empty({1, 2}, {5, 3}, DTO);
    auto B2 = Tensor::empty({}, {5, 3}, DTO);
    REQUIRE(utils::broadcastable(A2, B2));

    // Not broadcastable: batch-broadcastable but base shapes are different
    auto A3 = Tensor::empty({1, 2}, {5, 3}, DTO);
    auto B3 = Tensor::empty({1}, {1, 3}, DTO);
    REQUIRE(!utils::broadcastable(A3, B3));

    // Not broadcastable: batch shapes not broadcastable
    auto A4 = Tensor::empty({1, 2}, {5, 3}, DTO);
    auto B4 = Tensor::empty({3, 5}, {5, 3}, DTO);
    REQUIRE(!utils::broadcastable(A4, B4));

    auto A = Tensor::empty({1, 1, 1}, {5, 3}, DTO);
    auto B = Tensor::empty({5, 1, 2}, {5, 3}, DTO);
    auto C = Tensor::empty({2, 1}, {5, 3}, DTO);
    auto D = Tensor::empty({2, 2}, {5, 3}, DTO);
    REQUIRE(utils::broadcastable(A, B, C, D));

    auto E = Tensor::empty({3, 1, 1}, {5, 3}, DTO);
    REQUIRE(!utils::broadcastable(A, B, C, D, E));
  }

  SECTION("storage_size")
  {
    REQUIRE(utils::storage_size({}) == 1);
    REQUIRE(utils::storage_size({0}) == 0);
    REQUIRE(utils::storage_size({1}) == 1);
    REQUIRE(utils::storage_size({1, 2, 3}) == 6);
    REQUIRE(utils::storage_size({5, 1, 1}) == 5);
  }

  SECTION("add_shapes")
  {
    TensorShape s1 = {};
    TensorShape s2 = {2, 3};
    TensorShape s3 = {1, 2};
    TensorShape s4 = {12, 3};
    TensorShape s5 = {1, 2, 3};
    REQUIRE(utils::add_shapes(s1, s1) == s1);
    REQUIRE(utils::add_shapes(s1, s2) == s2);
    REQUIRE(utils::add_shapes(s3, s1, s4) == TensorShape{1, 2, 12, 3});
    REQUIRE(utils::add_shapes(s5, 3, 5, s2) == TensorShape{1, 2, 3, 3, 5, 2, 3});
  }

  SECTION("pad_prepend")
  {
    REQUIRE(utils::pad_prepend({3, 5}, 5, 1) == TensorShape{1, 1, 1, 3, 5});
    REQUIRE(utils::pad_prepend({3, 5}, 3, 3) == TensorShape{3, 3, 5});
    REQUIRE(utils::pad_prepend({3, 5}, 2) == TensorShape{3, 5});
  }
}
