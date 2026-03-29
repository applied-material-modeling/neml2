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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("SparseVector", "[equation_systems]")
{
  // A simple 2-variable, 1-group layout:
  //   var 0 "x": base_sizes={2}, intmd_sizes={}
  //   var 1 "y": base_sizes={1}, intmd_sizes={}
  const AxisLayout layout({{LabeledAxisAccessor("x"), LabeledAxisAccessor("y")}},
                          {TensorShape{}, TensorShape{}},
                          {TensorShape{2}, TensorShape{1}});

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

  SECTION("operator-")
  {
    SparseVector sv(layout);
    sv.tensors[0] = Tensor::create({1.0, -2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto neg = -sv;
    REQUIRE(neg.tensors[0].defined());
    REQUIRE_THAT(neg.tensors[0], test::allclose(Tensor::create({-1.0, 2.0}, 0, 0)));
    REQUIRE(!neg.tensors[1].defined());
  }

  SECTION("operator+")
  {
    SparseVector a(layout);
    SparseVector b(layout);
    a.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b.tensors[0] = Tensor::create({3.0, 4.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto c = a + b;
    REQUIRE(c.tensors[0].defined());
    REQUIRE_THAT(c.tensors[0], test::allclose(Tensor::create({4.0, 6.0}, 0, 0)));
    // a.tensors[1] is undefined, b.tensors[1] is defined: result should equal b.tensors[1]
    REQUIRE(c.tensors[1].defined());
    REQUIRE_THAT(c.tensors[1], test::allclose(b.tensors[1]));

    // Incompatible sizes should throw
    const AxisLayout layout1({{LabeledAxisAccessor("z")}}, {TensorShape{}}, {TensorShape{1}});
    SparseVector bad(layout1);
    REQUIRE_THROWS(a + bad);
  }

  SECTION("operator* scalar")
  {
    SparseVector sv(layout);
    sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto s = Scalar(2.0, sv.tensors[0].options());
    const auto left = s * sv;
    const auto right = sv * s;
    REQUIRE_THAT(left.tensors[0], test::allclose(Tensor::create({2.0, 4.0}, 0, 0)));
    REQUIRE_THAT(right.tensors[0], test::allclose(Tensor::create({2.0, 4.0}, 0, 0)));
    REQUIRE(!left.tensors[1].defined());
    REQUIRE(!right.tensors[1].defined());
  }

  SECTION("operator* inner product")
  {
    SparseVector a(layout);
    SparseVector b(layout);
    a.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    b.tensors[0] = Tensor::create({3.0, 4.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    // a.tensors[1] is defined, b.tensors[1] is not: the pair is skipped
    a.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    // 1*3 + 2*4 = 11
    const auto s = a * b;
    REQUIRE(s.item<double>() == 11.0);

    // Incompatible sizes should throw
    const AxisLayout layout1({{LabeledAxisAccessor("z")}}, {TensorShape{}}, {TensorShape{1}});
    SparseVector bad(layout1);
    REQUIRE_THROWS(a * bad);
  }

  SECTION("norm_sq")
  {
    SparseVector a(layout);
    a.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    a.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    // 1^2 + 2^2 + 5^2 = 30
    const auto ns = norm_sq(a);
    REQUIRE(ns.item<double>() == 30.0);
  }

  SECTION("norm")
  {
    SparseVector a(layout);
    a.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    a.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto n = norm(a);
    REQUIRE(n.item<double>() == std::sqrt(30.0));
  }

  SECTION("size")
  {
    SparseVector sv(layout);
    REQUIRE(sv.size() == 2);
  }

  SECTION("ngroup")
  {
    SECTION("one group")
    {
      SparseVector sv(layout);
      REQUIRE(sv.ngroup() == 1);
    }

    SECTION("two groups")
    {
      const AxisLayout layout2({{LabeledAxisAccessor("x")}, {LabeledAxisAccessor("y")}},
                               {TensorShape{}, TensorShape{}},
                               {TensorShape{2}, TensorShape{1}});
      SparseVector sv(layout2);
      REQUIRE(sv.ngroup() == 2);
    }
  }

  SECTION("group")
  {
    // A 2-variable, 2-group layout so that group(i) extracts a single variable each
    const AxisLayout layout2({{LabeledAxisAccessor("x")}, {LabeledAxisAccessor("y")}},
                             {TensorShape{}, TensorShape{}},
                             {TensorShape{2}, TensorShape{1}});
    SparseVector sv(layout2);
    sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
    sv.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    const auto g0 = sv.group(0);
    REQUIRE(g0.size() == 1);
    REQUIRE(g0.tensors[0].defined());
    REQUIRE_THAT(g0.tensors[0], test::allclose(Tensor::create({1.0, 2.0}, 0, 0)));

    const auto g1 = sv.group(1);
    REQUIRE(g1.size() == 1);
    REQUIRE(g1.tensors[0].defined());
    REQUIRE_THAT(g1.tensors[0], test::allclose(Tensor::create({5.0}, 0, 0)));
  }

  SECTION("assemble")
  {
    SECTION("IStructure::BLOCK")
    {
      SECTION("all tensors defined")
      {
        SparseVector sv(layout, SparseVector::IStructure::BLOCK);
        sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
        sv.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

        const auto assembled = sv.assemble();
        REQUIRE(assembled.base_dim() == 1);
        REQUIRE(assembled.base_sizes() == TensorShapeRef{3});
        REQUIRE_THAT(assembled, test::allclose(Tensor::create({1.0, 2.0, 5.0}, 0, 0)));
      }

      SECTION("undefined tensors are filled with zeros")
      {
        SparseVector sv(layout, SparseVector::IStructure::BLOCK);
        sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
        // sv.tensors[1] is undefined

        const auto assembled = sv.assemble();
        REQUIRE(assembled.base_sizes() == TensorShapeRef{3});
        REQUIRE_THAT(assembled, test::allclose(Tensor::create({1.0, 2.0, 0.0}, 0, 0)));
      }
    }

    SECTION("IStructure::DENSE")
    {
      // A layout with a non-trivial intermediate shape: intmd_sizes={2}, base_sizes={3}
      const AxisLayout layout_intmd(
          {{LabeledAxisAccessor("z")}}, {TensorShape{2}}, {TensorShape{3}});
      SparseVector sv(layout_intmd, SparseVector::IStructure::DENSE);
      sv.tensors[0] =
          Tensor::create({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/1);

      const auto assembled = sv.assemble();
      // intermediate and base dims are both folded into a single flat base dim
      REQUIRE(assembled.intmd_dim() == 0);
      REQUIRE(assembled.base_sizes() == TensorShapeRef{6}); // 2 * 3
    }
  }

  SECTION("disassemble")
  {
    SECTION("IStructure::BLOCK")
    {
      SparseVector sv(layout, SparseVector::IStructure::BLOCK);
      sv.tensors[0] = Tensor::create({1.0, 2.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);
      sv.tensors[1] = Tensor::create({5.0}, /*dynamic_dim=*/0, /*intmd_dim=*/0);

      const auto assembled = sv.assemble();
      SparseVector sv2(layout, SparseVector::IStructure::BLOCK);
      sv2.disassemble(assembled);

      REQUIRE_THAT(sv2.tensors[0], test::allclose(sv.tensors[0]));
      REQUIRE_THAT(sv2.tensors[1], test::allclose(sv.tensors[1]));
    }

    SECTION("IStructure::DENSE")
    {
      const AxisLayout layout_intmd(
          {{LabeledAxisAccessor("z")}}, {TensorShape{2}}, {TensorShape{3}});
      SparseVector sv(layout_intmd, SparseVector::IStructure::DENSE);
      sv.tensors[0] =
          Tensor::create({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, /*dynamic_dim=*/0, /*intmd_dim=*/1);

      const auto assembled = sv.assemble();
      SparseVector sv2(layout_intmd, SparseVector::IStructure::DENSE);
      sv2.disassemble(assembled);

      REQUIRE_THAT(sv2.tensors[0], test::allclose(sv.tensors[0]));
    }
  }
}
