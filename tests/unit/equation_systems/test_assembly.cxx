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

#include "neml2/equation_systems/assembly.h"
#include "neml2/tensors/Derivative.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("assembly", "[equation_systems]")
{
  SECTION("pop_intrsc_intmd_dim Tensor")
  {
    const auto t = Tensor::create({{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}},
                                   {{9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}}},
                                  /*dynamic_dim=*/0,
                                  /*intmd_dim=*/2);
    REQUIRE(t.intmd_sizes() == TensorShapeRef{2, 2});
    REQUIRE(t.base_sizes() == TensorShapeRef{4});

    const auto popped = pop_intrsc_intmd_dim(t, 1);
    REQUIRE(popped.intmd_sizes() == TensorShapeRef{2});
    REQUIRE(popped.base_sizes() == TensorShapeRef{2, 4});

    const auto roundtrip = push_intrsc_intmd_dim(popped, 1);
    REQUIRE_THAT(roundtrip, test::allclose(t));
  }

  SECTION("push_intrsc_intmd_dim Tensor")
  {
    const auto t = Tensor::create(
        {{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}},
        /*dynamic_dim=*/0,
        /*intmd_dim=*/1);
    REQUIRE(t.intmd_sizes() == TensorShapeRef{2});
    REQUIRE(t.base_sizes() == TensorShapeRef{3, 2});

    const auto pushed = push_intrsc_intmd_dim(t, 1);
    REQUIRE(pushed.intmd_sizes() == TensorShapeRef{2, 3});
    REQUIRE(pushed.base_sizes() == TensorShapeRef{2});

    const auto roundtrip = pop_intrsc_intmd_dim(pushed, 1);
    REQUIRE_THAT(roundtrip, test::allclose(t));
  }

  SECTION("pop_intrsc_intmd_dim Derivative")
  {
    Derivative<1> deriv(
        1, {1, 1}, {TensorShape{2}, TensorShape{2}}, {TensorShape{}, TensorShape{}}, "y", {"x"});
    deriv = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/1);

    const auto popped = pop_intrsc_intmd_dim(deriv);
    REQUIRE(popped.intmd_dim() == 0);
    REQUIRE(popped.base_sizes() == TensorShapeRef{2, 2});
    REQUIRE(popped.base_index({0, 0}).item<double>() == 1.0);
    REQUIRE(popped.base_index({1, 1}).item<double>() == 2.0);
    REQUIRE(popped.base_index({0, 1}).item<double>() == 0.0);
    REQUIRE(popped.base_index({1, 0}).item<double>() == 0.0);
  }

  SECTION("pop/push_intrsc_intmd_dim template roundtrip")
  {
    const auto t = Tensor::create(
        {{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}, {{8.0, 9.0}, {10.0, 11.0}}},
         {{{12.0, 13.0}, {14.0, 15.0}},
          {{16.0, 17.0}, {18.0, 19.0}},
          {{20.0, 21.0}, {22.0, 23.0}}}},
        /*dynamic_dim=*/0,
        /*intmd_dim=*/2);
    const std::array<std::size_t, 2> intrsc_dims = {1, 1};
    const std::array<TensorShape, 2> base_shapes = {TensorShape{2}, TensorShape{2}};
    const auto popped = pop_intrsc_intmd_dim<2>(t, intrsc_dims, base_shapes, "t");
    const auto roundtrip = push_intrsc_intmd_dim<2>(popped, intrsc_dims, base_shapes, "t");
    REQUIRE_THAT(roundtrip, test::allclose(t));
  }

  SECTION("to_assembly/from_assembly N==1")
  {
    SECTION("roundtrip")
    {
      const auto t =
          Tensor::create({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/1);
      const auto assembly = to_assembly<1>(t, {TensorShape{2}}, {TensorShape{3}});
      REQUIRE(assembly.intmd_dim() == 0);
      REQUIRE(assembly.base_sizes() == TensorShapeRef{6});

      const auto back = from_assembly<1>(assembly, {TensorShape{2}}, {TensorShape{3}});
      REQUIRE_THAT(back, test::allclose(t));
    }

    SECTION("broadcast")
    {
      const auto t = Tensor::create({{1.0, 2.0, 3.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/1);
      const auto assembly = to_assembly<1>(t, {TensorShape{2}}, {TensorShape{3}});
      const auto back = from_assembly<1>(assembly, {TensorShape{2}}, {TensorShape{3}});
      REQUIRE(back.intmd_sizes() == TensorShapeRef{2});
      REQUIRE(back.base_sizes() == TensorShapeRef{3});
      REQUIRE(back.intmd_index({0}).base_index({0}).item<double>() == 1.0);
      REQUIRE(back.intmd_index({1}).base_index({0}).item<double>() == 1.0);
    }
  }

  SECTION("to_assembly/from_assembly N==2")
  {
    const auto t = Tensor::create(
        {{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}, {{8.0, 9.0}, {10.0, 11.0}}},
         {{{12.0, 13.0}, {14.0, 15.0}},
          {{16.0, 17.0}, {18.0, 19.0}},
          {{20.0, 21.0}, {22.0, 23.0}}}},
        /*dynamic_dim=*/0,
        /*intmd_dim=*/2);
    const auto assembly =
        to_assembly<2>(t, {TensorShape{2}, TensorShape{3}}, {TensorShape{2}, TensorShape{2}});
    REQUIRE(assembly.intmd_dim() == 0);
    REQUIRE(assembly.base_sizes() == TensorShapeRef{4, 6});

    const auto back = from_assembly<2>(
        assembly, {TensorShape{2}, TensorShape{3}}, {TensorShape{2}, TensorShape{2}});
    REQUIRE_THAT(back, test::allclose(t));
  }
}
