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

#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("SparseMatrix", "[equation_systems]")
{
  // A 2×2 block matrix in a single row group and single column group:
  //   row_layout: "p" (base={2}), "q" (base={1})
  //   col_layout: "x" (base={3}), "y" (base={1})
  //
  // The four blocks have base shapes:
  //   [0][0] "p"x"x": {2,3}   [0][1] "p"x"y": {2,1}
  //   [1][0] "q"x"x": {1,3}   [1][1] "q"x"y": {1,1}
  //
  // Fully assembled (assemble_intmd=false), the result is a {3,4} matrix:
  //   [[ 1,  2,  3,  7],
  //    [ 4,  5,  6,  8],
  //    [ 9, 10, 11, 12]]
  const AxisLayout row_layout({{LabeledAxisAccessor("p"), LabeledAxisAccessor("q")}},
                              {TensorShape{}, TensorShape{}},
                              {TensorShape{2}, TensorShape{1}},
                              {AxisLayout::IStructure::DENSE});
  const AxisLayout col_layout({{LabeledAxisAccessor("x"), LabeledAxisAccessor("y")}},
                              {TensorShape{}, TensorShape{}},
                              {TensorShape{3}, TensorShape{1}},
                              {AxisLayout::IStructure::DENSE});

  const auto t00 = Tensor::create({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                  /*dynamic_dim=*/0,
                                  /*intmd_dim=*/0);                                         // {2,3}
  const auto t01 = Tensor::create({{7.0}, {8.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/0);      // {2,1}
  const auto t10 = Tensor::create({{9.0, 10.0, 11.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/0); // {1,3}
  const auto t11 = Tensor::create({{12.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/0);            // {1,1}

  SECTION("SparseMatrix(row_layout, col_layout)")
  {
    // 2-arg constructor must pre-allocate nrow inner vectors, each of size ncol.
    SparseMatrix sm(row_layout, col_layout);
    REQUIRE(sm.tensors.size() == 2);    // nrow inner vectors
    REQUIRE(sm.tensors[0].size() == 2); // each sized to ncol
    REQUIRE(sm.tensors[1].size() == 2);
    REQUIRE(!sm.tensors[0][0].defined()); // all entries start undefined
    REQUIRE(!sm.tensors[0][1].defined());
    REQUIRE(!sm.tensors[1][0].defined());
    REQUIRE(!sm.tensors[1][1].defined());
  }

  SECTION("SparseMatrix(row_layout, col_layout, tensors)")
  {
    SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
    REQUIRE(sm.tensors.size() == 2);
    REQUIRE(sm.tensors[0].size() == 2);
    REQUIRE(sm.tensors[1].size() == 2);
    REQUIRE_THAT(sm.tensors[0][0], test::allclose(t00));
    REQUIRE_THAT(sm.tensors[0][1], test::allclose(t01));
    REQUIRE_THAT(sm.tensors[1][0], test::allclose(t10));
    REQUIRE_THAT(sm.tensors[1][1], test::allclose(t11));
  }

  SECTION("options")
  {
    SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
    const auto opts = sm.options();
    REQUIRE(opts.dtype() == t00.options().dtype());
    REQUIRE(opts.device() == t00.options().device());
  }

  SECTION("tensors")
  {
    SparseMatrix sm(row_layout, col_layout, {{t00, {}}, {{}, {}}});
    REQUIRE(sm.tensors[0][0].defined());
    REQUIRE(!sm.tensors[0][1].defined());
    REQUIRE(!sm.tensors[1][0].defined());
    REQUIRE(!sm.tensors[1][1].defined());
  }

  SECTION("group")
  {
    // 2-group row and 2-group col layouts so each block contains exactly one variable pair
    const AxisLayout row_layout2({{LabeledAxisAccessor("p")}, {LabeledAxisAccessor("q")}},
                                 {TensorShape{}, TensorShape{}},
                                 {TensorShape{2}, TensorShape{1}},
                                 {AxisLayout::IStructure::DENSE, AxisLayout::IStructure::DENSE});
    const AxisLayout col_layout2({{LabeledAxisAccessor("x")}, {LabeledAxisAccessor("y")}},
                                 {TensorShape{}, TensorShape{}},
                                 {TensorShape{3}, TensorShape{1}},
                                 {AxisLayout::IStructure::DENSE, AxisLayout::IStructure::DENSE});
    SparseMatrix sm(row_layout2, col_layout2, {{t00, t01}, {t10, t11}});

    const auto g00 = sm.group(0, 0);
    REQUIRE(g00.tensors[0][0].defined());
    REQUIRE_THAT(g00.tensors[0][0], test::allclose(t00));

    const auto g01 = sm.group(0, 1);
    REQUIRE_THAT(g01.tensors[0][0], test::allclose(t01));

    const auto g10 = sm.group(1, 0);
    REQUIRE_THAT(g10.tensors[0][0], test::allclose(t10));

    const auto g11 = sm.group(1, 1);
    REQUIRE_THAT(g11.tensors[0][0], test::allclose(t11));
  }
}
