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
#include "neml2/tensors/list_tensors.h"

using namespace neml2;

TEST_CASE("list_tensors", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape
  Size l1 = 6;                  // First list size
  Size l2 = 4;                  // Second list size

  TensorShape BS = {3, 3}; // Base shape

  SECTION("Base size same as input sizes")
  {

    // A list of R2s
    auto a = R2::fill(1.0, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l1}));

    // Another list of R2s
    auto b = R2::fill(1.0, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l2}));

    // A standard, unlisted R2
    auto c = R2::fill(3.0, DTO).batch_expand(B);

    auto binary_operator = [](const R2 & a, const R2 & b) { return a * b; };

    SECTION("outer product, first is a list")
    {
      REQUIRE(list_derivative_outer_product_a(binary_operator, a, c).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{l1}, BS));
    }

    SECTION("outer product, second is a list")
    {
      REQUIRE(list_derivative_outer_product_b(binary_operator, c, a).sizes() ==
              utils::add_shapes(B, BS, neml2::TensorShape{l1}));
    }

    SECTION("outer product, both are lists")
    {
      REQUIRE(list_derivative_outer_product_ab(binary_operator, a, b).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{l1}, BS, neml2::TensorShape{l2}));
    }
  }

  SECTION("Base size same as ones of input sizes")
  {
    // A list of R2s
    auto a_list = R2::fill(2.0, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l1}));

    // A standard R2
    auto a_nolist = R2::fill(2.0, DTO).batch_expand(B);

    // A list of scalars
    auto b_list = Scalar(3.0, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l2}));

    // A standard scalar
    auto b_nolist = Scalar(3.0, DTO).batch_expand(B);

    auto binary_operator = [](const auto & a, const auto & b) { return a * b; };

    SECTION("outer product, list x not")
    {
      REQUIRE(list_derivative_outer_product_a(binary_operator, a_list, b_nolist).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{l1}, neml2::TensorShape{3, 3}));
    }

    SECTION("outer product, not x list")
    {
      REQUIRE(list_derivative_outer_product_b(binary_operator, a_nolist, b_list).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{3, 3}, neml2::TensorShape{l2}));
    }

    SECTION("outer product, list x list")
    {
      REQUIRE(list_derivative_outer_product_ab(binary_operator, a_list, b_list).sizes() ==
              utils::add_shapes(
                  B, neml2::TensorShape{l1}, neml2::TensorShape{3, 3}, neml2::TensorShape{l2}));
    }
  }

  SECTION("None of the sizes are the same")
  {
    // A list of R2s
    auto b_list = R2::fill(2.0, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l2}));

    // A standard R2
    auto b_nolist = R2::fill(2.0, DTO).batch_expand(B);

    // A list of Rots
    auto a_list =
        Rot::fill(1.2, 3.1, -2.1, DTO).batch_expand(utils::add_shapes(B, neml2::TensorShape{l1}));

    // A standard Rot
    auto a_nolist = Rot::fill(1.2, 3.1, -2.1, DTO).batch_expand(B);

    auto binary_operator = [](const Rot & a, const R2 & b) { return b.drotate(a); };

    SECTION("outer product, list x not")
    {
      REQUIRE(list_derivative_outer_product_a(binary_operator, a_list, b_nolist).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{l1, 3}, neml2::TensorShape{3, 3}));
    }

    SECTION("outer product, not x list")
    {
      REQUIRE(list_derivative_outer_product_b(binary_operator, a_nolist, b_list).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{3}, neml2::TensorShape{l2, 3, 3}));
    }

    SECTION("outer product, list x list")
    {
      REQUIRE(list_derivative_outer_product_ab(binary_operator, a_list, b_list).sizes() ==
              utils::add_shapes(B, neml2::TensorShape{l1, 3}, neml2::TensorShape{l2, 3, 3}));
    }
  }
}
