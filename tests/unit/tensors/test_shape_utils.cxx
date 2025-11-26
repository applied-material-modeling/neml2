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
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/Tensor.h"

using namespace neml2;

TEST_CASE("shape_utils", "[tensors]")
{
  SECTION("normalize_dim")
  {
    REQUIRE(utils::normalize_dim(0, 0, 3) == 0);
    REQUIRE(utils::normalize_dim(1, 2, 5) == 3);
    REQUIRE(utils::normalize_dim(-1, 1, 4) == 3);
    REQUIRE(utils::normalize_dim(-3, 1, 4) == 1);
#ifndef NDEBUG
    REQUIRE_THROWS(utils::normalize_dim(-4, 1, 4),
                   Catch::Matchers::ContainsSubstring("The dimension -4 is out of range"));
    REQUIRE_THROWS(utils::normalize_dim(5, 1, 4),
                   Catch::Matchers::ContainsSubstring("The dimension 5 is out of range"));
#endif
  }

  SECTION("normalize_itr")
  {
    REQUIRE(utils::normalize_itr(0, 0, 3) == 0);
    REQUIRE(utils::normalize_itr(3, 2, 5) == 5);
    REQUIRE(utils::normalize_itr(-1, 1, 4) == 4);
    REQUIRE(utils::normalize_itr(-4, 1, 4) == 1);
#ifndef NDEBUG
    REQUIRE_THROWS(utils::normalize_itr(-5, 1, 4),
                   Catch::Matchers::ContainsSubstring("The dimension -5 is out of range"));
    REQUIRE_THROWS(utils::normalize_itr(4, 1, 4),
                   Catch::Matchers::ContainsSubstring("The dimension 4 is out of range"));
#endif
  }

  SECTION("sizes_broadcastable")
  {
    REQUIRE(utils::sizes_broadcastable(
        TensorShapeRef{}, TensorShapeRef{1, 2, 3}, TensorShapeRef{4, 1, 3}));
    REQUIRE(utils::sizes_broadcastable(
        TensorShapeRef{5, 1, 1}, TensorShapeRef{1, 2, 1}, TensorShapeRef{1, 1, 3}));
    REQUIRE(utils::sizes_broadcastable(TensorShapeRef{1, 2, 3}, TensorShapeRef{2, 3}));
    REQUIRE(!utils::sizes_broadcastable(TensorShapeRef{2, 3}, TensorShapeRef{3, 2}));
    REQUIRE(!utils::sizes_broadcastable(TensorShapeRef{1, 2}, TensorShapeRef{1, 2, 3}));
  }

  SECTION("broadcastable")
  {
    auto A = Tensor::empty({1, 1, 1}, {1, 3}, {5, 3});
    auto B = Tensor::empty({5, 1, 2}, {2, 1}, {1, 1, 3});
    auto C = Tensor::empty({2, 1}, {1}, {5, 1});
    auto D = Tensor::empty({2, 2}, {3}, {1});
    REQUIRE(utils::broadcastable(A, B, C, D));

    auto E = Tensor::empty({3}, {}, {});
    REQUIRE(!utils::broadcastable(A, B, C, D, E));

    auto F = Tensor::empty({}, {5, 1}, {});
    REQUIRE(!utils::broadcastable(A, B, C, D, F));

    auto G = Tensor::empty({}, {}, {2, 3});
    REQUIRE(!utils::broadcastable(A, B, C, D, G));
  }

  SECTION("dynamic_broadcastable")
  {
    auto A = Tensor::empty({1, 1, 1}, {2}, {});
    auto B = Tensor::empty({5, 1, 2}, {3}, {2, 2});
    auto C = Tensor::empty({2, 1}, {1}, {1, 3});
    auto D = Tensor::empty({2, 2}, {4}, {7});
    REQUIRE(utils::dynamic_broadcastable(A, B, C, D));

    auto E = Tensor::empty({3}, {}, {});
    REQUIRE(!utils::dynamic_broadcastable(A, B, C, D, E));
  }

  SECTION("intmd_broadcastable")
  {
    auto A = Tensor::empty({2}, {1, 3}, {2});
    auto B = Tensor::empty({1}, {2, 1}, {});
    auto C = Tensor::empty({4}, {1}, {1, 1});
    auto D = Tensor::empty({3}, {3}, {3});
    REQUIRE(utils::intmd_broadcastable(A, B, C, D));

    auto E = Tensor::empty({}, {2}, {});
    REQUIRE(!utils::intmd_broadcastable(A, B, C, D, E));
  }

  SECTION("base_broadcastable")
  {
    auto A = Tensor::empty({2}, {2}, {5, 3});
    auto B = Tensor::empty({1}, {3}, {1, 1, 3});
    auto C = Tensor::empty({4}, {1}, {5, 1});
    auto D = Tensor::empty({3}, {4}, {1});
    REQUIRE(utils::base_broadcastable(A, B, C, D));

    auto E = Tensor::empty({}, {}, {7});
    REQUIRE(!utils::base_broadcastable(A, B, C, D, E));
  }

  SECTION("broadcast_dynamic_dim")
  {
    TensorShape a = {};
    TensorShape b = {1, 2};
    TensorShape c = {3};
    TensorShape d = {4, 5, 6};

    // Create some tensors with the above batch shapes, the base shapes should not matter.
    auto A = Tensor::empty(a, {}, {5, 3});
    auto B = Tensor::empty(b, {}, {1, 2});
    auto C = Tensor::empty(c, {}, {});
    auto D = Tensor::empty(d, {}, {3, 5, 6});

    REQUIRE(utils::broadcast_dynamic_dim(A) == 0);
    REQUIRE(utils::broadcast_dynamic_dim(A, B) == 2);
    REQUIRE(utils::broadcast_dynamic_dim(A, B, C) == 2);
    REQUIRE(utils::broadcast_dynamic_dim(A, B, C, D) == 3);
  }

  SECTION("broadcast_intmd_dim")
  {
    TensorShape a = {};
    TensorShape b = {1, 2};
    TensorShape c = {3};
    TensorShape d = {4, 5, 6};

    // Create some tensors with the above batch shapes, the base shapes should not matter.
    auto A = Tensor::empty({}, a, {5, 3});
    auto B = Tensor::empty({}, b, {1, 2});
    auto C = Tensor::empty({}, c, {});
    auto D = Tensor::empty({}, d, {3, 5, 6});

    REQUIRE(utils::broadcast_intmd_dim(A) == 0);
    REQUIRE(utils::broadcast_intmd_dim(A, B) == 2);
    REQUIRE(utils::broadcast_intmd_dim(A, B, C) == 2);
    REQUIRE(utils::broadcast_intmd_dim(A, B, C, D) == 3);
  }

  SECTION("broadcast_base_dim")
  {
    TensorShape a = {};
    TensorShape b = {1, 2};
    TensorShape c = {3};
    TensorShape d = {4, 5, 6};

    // Create some tensors with the above batch shapes, the base shapes should not matter.
    auto A = Tensor::empty({}, {5, 3}, a);
    auto B = Tensor::empty({}, {1, 2}, b);
    auto C = Tensor::empty({}, {}, c);
    auto D = Tensor::empty({}, {3, 5, 6}, d);

    REQUIRE(utils::broadcast_base_dim(A) == 0);
    REQUIRE(utils::broadcast_base_dim(A, B) == 2);
    REQUIRE(utils::broadcast_base_dim(A, B, C) == 2);
    REQUIRE(utils::broadcast_base_dim(A, B, C, D) == 3);
  }

  SECTION("broadcast_sizes")
  {
    TensorShape a = {};
    TensorShape b = {1, 2};
    TensorShape c = {5, 1};
    TensorShape d = {4, 1, 2};
    REQUIRE(utils::broadcast_sizes(a, b) == TensorShape{1, 2});
    REQUIRE(utils::broadcast_sizes(a, b, c) == TensorShape{5, 2});
    REQUIRE(utils::broadcast_sizes(a, b, c, d) == TensorShape{4, 5, 2});
  }

  SECTION("numel")
  {
    REQUIRE(utils::numel({}) == 1);
    REQUIRE(utils::numel({0}) == 0);
    REQUIRE(utils::numel({1}) == 1);
    REQUIRE(utils::numel({1, 2, 3}) == 6);
    REQUIRE(utils::numel({5, 1, 1}) == 5);
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
