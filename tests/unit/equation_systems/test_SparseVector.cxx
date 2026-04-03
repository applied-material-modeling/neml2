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

#include "neml2/equation_systems/SparseVector.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("SparseVector", "[equation_systems]")
{
  // A simple 2-variable, 1-group layout:
  //   var 0 "x": base_sizes={2}, intmd_sizes={}
  //   var 1 "y": base_sizes={1}, intmd_sizes={}
  const AxisLayout layout({{VariableName("x"), VariableName("y")}},
                          {TensorShape{}, TensorShape{}},
                          {TensorShape{2}, TensorShape{1}},
                          {AxisLayout::IStructure::DENSE});

  SECTION("SparseVector(layout, tensors)")
  {
    std::vector<Tensor> ts = {Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0),
                              Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0)};
    SparseVector sv(layout, ts);
    REQUIRE(sv.tensors.size() == 2);
    REQUIRE_THAT(sv.tensors[0], test::allclose(Tensor::create({1.0, 2.0}, 0, 0)));
    REQUIRE_THAT(sv.tensors[1], test::allclose(Tensor::create({5.0}, 0, 0)));
  }

  SECTION("options")
  {
    SparseVector sv(layout);
    sv.tensors[1] = Tensor::create({3.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    const auto opts = sv.options();
    REQUIRE(opts.dtype() == sv.tensors[1].options().dtype());
    REQUIRE(opts.device() == sv.tensors[1].options().device());
  }

  SECTION("tensors")
  {
    SparseVector sv(layout);
    sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    REQUIRE(sv.tensors.size() == 2);
    REQUIRE(sv.tensors[0].defined());
    REQUIRE_THAT(sv.tensors[0], test::allclose(Tensor::create({1.0, 2.0}, 0, 0)));
    REQUIRE(!sv.tensors[1].defined());
  }

  SECTION("group")
  {
    // A 2-variable, 2-group layout so that group(i) extracts a single variable each
    const AxisLayout layout2({{VariableName("x")}, {VariableName("y")}},
                             {TensorShape{}, TensorShape{}},
                             {TensorShape{2}, TensorShape{1}},
                             {AxisLayout::IStructure::DENSE, AxisLayout::IStructure::DENSE});
    SparseVector sv(layout2);
    sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    sv.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto g0 = sv.group(0);
    REQUIRE(g0.tensors[0].defined());
    REQUIRE_THAT(g0.tensors[0], test::allclose(Tensor::create({1.0, 2.0}, 0, 0)));

    const auto g1 = sv.group(1);
    REQUIRE(g1.tensors[0].defined());
    REQUIRE_THAT(g1.tensors[0], test::allclose(Tensor::create({5.0}, 0, 0)));
  }
}
