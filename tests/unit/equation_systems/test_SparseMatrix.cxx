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
                              {TensorShape{2}, TensorShape{1}});
  const AxisLayout col_layout({{LabeledAxisAccessor("x"), LabeledAxisAccessor("y")}},
                              {TensorShape{}, TensorShape{}},
                              {TensorShape{3}, TensorShape{1}});

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

  SECTION("nrow")
  {
    SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
    REQUIRE(sm.nrow() == 2);
  }

  SECTION("ncol")
  {
    SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
    REQUIRE(sm.ncol() == 2);
  }

  SECTION("size")
  {
    // size() returns tensors.size(), i.e. the number of row variables
    SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
    REQUIRE(sm.size() == sm.nrow());
  }

  SECTION("row_ngroup")
  {
    SECTION("one group")
    {
      SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
      REQUIRE(sm.row_ngroup() == 1);
    }

    SECTION("two groups")
    {
      const AxisLayout row_layout2({{LabeledAxisAccessor("p")}, {LabeledAxisAccessor("q")}},
                                   {TensorShape{}, TensorShape{}},
                                   {TensorShape{2}, TensorShape{1}});
      SparseMatrix sm(row_layout2, col_layout, {{t00, t01}, {t10, t11}});
      REQUIRE(sm.row_ngroup() == 2);
    }
  }

  SECTION("col_ngroup")
  {
    SECTION("one group")
    {
      SparseMatrix sm(row_layout, col_layout, {{t00, t01}, {t10, t11}});
      REQUIRE(sm.col_ngroup() == 1);
    }

    SECTION("two groups")
    {
      const AxisLayout col_layout2({{LabeledAxisAccessor("x")}, {LabeledAxisAccessor("y")}},
                                   {TensorShape{}, TensorShape{}},
                                   {TensorShape{3}, TensorShape{1}});
      SparseMatrix sm(row_layout, col_layout2, {{t00, t01}, {t10, t11}});
      REQUIRE(sm.col_ngroup() == 2);
    }
  }

  SECTION("group")
  {
    // 2-group row and 2-group col layouts so each block contains exactly one variable pair
    const AxisLayout row_layout2({{LabeledAxisAccessor("p")}, {LabeledAxisAccessor("q")}},
                                 {TensorShape{}, TensorShape{}},
                                 {TensorShape{2}, TensorShape{1}});
    const AxisLayout col_layout2({{LabeledAxisAccessor("x")}, {LabeledAxisAccessor("y")}},
                                 {TensorShape{}, TensorShape{}},
                                 {TensorShape{3}, TensorShape{1}});
    SparseMatrix sm(row_layout2, col_layout2, {{t00, t01}, {t10, t11}});

    const auto g00 = sm.group(0, 0);
    REQUIRE(g00.nrow() == 1);
    REQUIRE(g00.ncol() == 1);
    REQUIRE(g00.tensors[0][0].defined());
    REQUIRE_THAT(g00.tensors[0][0], test::allclose(t00));

    const auto g01 = sm.group(0, 1);
    REQUIRE(g01.nrow() == 1);
    REQUIRE(g01.ncol() == 1);
    REQUIRE_THAT(g01.tensors[0][0], test::allclose(t01));

    const auto g10 = sm.group(1, 0);
    REQUIRE(g10.nrow() == 1);
    REQUIRE(g10.ncol() == 1);
    REQUIRE_THAT(g10.tensors[0][0], test::allclose(t10));

    const auto g11 = sm.group(1, 1);
    REQUIRE(g11.nrow() == 1);
    REQUIRE(g11.ncol() == 1);
    REQUIRE_THAT(g11.tensors[0][0], test::allclose(t11));
  }

  SECTION("assemble")
  {
    SECTION("IStructure::BLOCK_DIAGONAL")
    {
      SECTION("all tensors defined")
      {
        SparseMatrix sm(row_layout,
                        col_layout,
                        {{t00, t01}, {t10, t11}},
                        SparseMatrix::IStructure::BLOCK_DIAGONAL);
        const auto assembled = sm.assemble();
        REQUIRE(assembled.base_dim() == 2);
        REQUIRE(assembled.base_sizes() == TensorShapeRef{3, 4}); // (2+1) x (3+1)
        REQUIRE_THAT(assembled,
                     test::allclose(Tensor::create(
                         {{1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}},
                         /*dynamic_dim=*/0,
                         /*intmd_dim=*/0)));
      }

      SECTION("undefined tensors are filled with zeros")
      {
        // only [0][0] is defined; the rest are zeros
        SparseMatrix sm(row_layout,
                        col_layout,
                        {{t00, {}}, {{}, {}}},
                        SparseMatrix::IStructure::BLOCK_DIAGONAL);
        const auto assembled = sm.assemble();
        REQUIRE(assembled.base_sizes() == TensorShapeRef{3, 4});
        REQUIRE_THAT(assembled,
                     test::allclose(Tensor::create(
                         {{1.0, 2.0, 3.0, 0.0}, {4.0, 5.0, 6.0, 0.0}, {0.0, 0.0, 0.0, 0.0}},
                         /*dynamic_dim=*/0,
                         /*intmd_dim=*/0)));
      }
    }

    SECTION("IStructure::DENSE")
    {
      // 1×1 matrix with non-trivial intermediate shapes:
      //   row "p": intmd_sizes={2}, base_sizes={2}
      //   col "x": intmd_sizes={3}, base_sizes={2}
      // The block tensor has intmd_dim=2, base_dim=2, total shape {2,3,2,2}.
      const AxisLayout row_intmd({{LabeledAxisAccessor("p")}}, {TensorShape{2}}, {TensorShape{2}});
      const AxisLayout col_intmd({{LabeledAxisAccessor("x")}}, {TensorShape{3}}, {TensorShape{2}});
      const auto block = Tensor::create(
          {{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}, {{8.0, 9.0}, {10.0, 11.0}}},
           {{{12.0, 13.0}, {14.0, 15.0}},
            {{16.0, 17.0}, {18.0, 19.0}},
            {{20.0, 21.0}, {22.0, 23.0}}}},
          /*dynamic_dim=*/0,
          /*intmd_dim=*/2);
      SparseMatrix sm(row_intmd, col_intmd, {{block}});

      const auto assembled = sm.assemble();
      // intermediate and base dims are both folded into flat base dims
      REQUIRE(assembled.intmd_dim() == 0);
      REQUIRE(assembled.base_sizes() == TensorShapeRef{4, 6}); // (2*2) x (3*2)
    }
  }

  SECTION("disassemble")
  {
    SECTION("IStructure::BLOCK_DIAGONAL")
    {
      SparseMatrix sm(row_layout,
                      col_layout,
                      {{t00, t01}, {t10, t11}},
                      SparseMatrix::IStructure::BLOCK_DIAGONAL);
      const auto assembled = sm.assemble();

      // Use the 2-arg constructor (layout-only) to create the output shell — this is
      // exactly the pattern used by DenseLU::solve. A bug in the 2-arg constructor
      // (e.g., wrong inner-vector sizing) would be caught here.
      SparseMatrix sm2(row_layout, col_layout, SparseMatrix::IStructure::BLOCK_DIAGONAL);
      sm2.disassemble(assembled);

      REQUIRE_THAT(sm2.tensors[0][0], test::allclose(t00));
      REQUIRE_THAT(sm2.tensors[0][1], test::allclose(t01));
      REQUIRE_THAT(sm2.tensors[1][0], test::allclose(t10));
      REQUIRE_THAT(sm2.tensors[1][1], test::allclose(t11));
    }

    SECTION("IStructure::DENSE")
    {
      const AxisLayout row_intmd({{LabeledAxisAccessor("p")}}, {TensorShape{2}}, {TensorShape{2}});
      const AxisLayout col_intmd({{LabeledAxisAccessor("x")}}, {TensorShape{3}}, {TensorShape{2}});
      const auto block = Tensor::create(
          {{{{0.0, 1.0}, {2.0, 3.0}}, {{4.0, 5.0}, {6.0, 7.0}}, {{8.0, 9.0}, {10.0, 11.0}}},
           {{{12.0, 13.0}, {14.0, 15.0}},
            {{16.0, 17.0}, {18.0, 19.0}},
            {{20.0, 21.0}, {22.0, 23.0}}}},
          /*dynamic_dim=*/0,
          /*intmd_dim=*/2);
      SparseMatrix sm(row_intmd, col_intmd, {{block}}, SparseMatrix::IStructure::DENSE);

      const auto assembled = sm.assemble();
      // Also use 2-arg constructor here to cover the intermediate-shapes path
      SparseMatrix sm2(row_intmd, col_intmd, SparseMatrix::IStructure::DENSE);
      sm2.disassemble(assembled);

      REQUIRE_THAT(sm2.tensors[0][0], test::allclose(block));
    }
  }

  SECTION("operator-")
  {
    SparseMatrix sm(
        row_layout, col_layout, {{t00, {}}, {t10, {}}}, SparseMatrix::IStructure::DENSE);
    const auto neg = -sm;

    // defined entries are negated
    REQUIRE(neg.tensors[0][0].defined());
    REQUIRE_THAT(neg.tensors[0][0], test::allclose(-t00));
    REQUIRE(neg.tensors[1][0].defined());
    REQUIRE_THAT(neg.tensors[1][0], test::allclose(-t10));

    // undefined entries stay undefined
    REQUIRE(!neg.tensors[0][1].defined());
    REQUIRE(!neg.tensors[1][1].defined());
  }
}
