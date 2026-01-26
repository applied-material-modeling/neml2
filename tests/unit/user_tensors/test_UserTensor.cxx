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

#include "neml2/neml2.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/SFR3.h"

using namespace neml2;

TEST_CASE("UserTensor", "[user_tensors]")
{
  SECTION("load correctly")
  {
    auto factory = load_input("user_tensors/test_UserTensor.i");

    const auto a = factory->get_object<Tensor>("Tensors", "a");
    const auto a_correct = Tensor::create({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
    REQUIRE(at::allclose(*a, a_correct));
    REQUIRE(a->batch_sizes() == TensorShape{2});
    REQUIRE(a->base_sizes() == TensorShape{2, 3});

    const auto b = factory->get_object<Tensor>("Tensors", "b");
    const auto b_correct = Tensor::create({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
    REQUIRE(at::allclose(*b, b_correct));
    REQUIRE(b->batch_sizes() == TensorShape{});
    REQUIRE(b->base_sizes() == TensorShape{2, 2, 3});

    const auto c = factory->get_object<Tensor>("Tensors", "c");
    const auto c_correct = Tensor::create({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
    REQUIRE(at::allclose(*c, c_correct));
    REQUIRE(c->batch_sizes() == TensorShape{2, 2, 3});
    REQUIRE(c->base_sizes() == TensorShape{});

    const auto d = factory->get_object<Tensor>("Tensors", "d");
    const auto d_correct = Tensor::create({{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
    REQUIRE(at::allclose(*d, d_correct));
    REQUIRE(d->dynamic_sizes() == TensorShape{2});
    REQUIRE(d->intmd_sizes() == TensorShape{3});
    REQUIRE(d->base_sizes() == TensorShape{2});

    const auto e = factory->get_object<SFR3>("Tensors", "e");
    const auto e_correct = Tensor::create(
        {{{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}},
         {{{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}, {-1, -2, -3}, {-4, -5, -6}, {-7, -8, -9}}}}},
        2,
        1);
    REQUIRE(at::allclose(*e, e_correct));
    REQUIRE(e->dynamic_sizes() == TensorShape{2, 1});
    REQUIRE(e->intmd_sizes() == TensorShape{1});
    REQUIRE(e->base_sizes() == TensorShape{6, 3});
  }

  SECTION("errors")
  {
    auto factory = load_input("user_tensors/test_UserTensor_error.i");

    REQUIRE_THROWS_WITH(
        factory->get_object<Tensor>("Tensors", "a"),
        Catch::Matchers::ContainsSubstring("Intermediate dimension 2 must be less than or equal to "
                                           "the number of batch dimensions 1"));
  }
}
