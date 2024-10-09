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

#include "neml2/tensors/LabeledTensor.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"

using namespace neml2;

TEST_CASE("LabeledTensor", "[tensors]")
{
  Size nbatch = 10;

  LabeledAxis info1;
  info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SR2>("second");
  info2.setup_layout();

  SECTION("copy constructor")
  {
    auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
    auto B = A;
    B.base_index_put_({"third", "second"}, Tensor::ones({1, 6}).batch_expand(nbatch));
    REQUIRE(torch::allclose(A, B));
    REQUIRE(A.axis(0) == B.axis(0));
    REQUIRE(A.axis(1) == B.axis(1));
  }

  SECTION("copy assignment operator")
  {
    auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
    LabeledMatrix B;
    B = A;
    B.base_index_put_({"third", "second"}, Tensor::ones({1, 6}).batch_expand(nbatch));
    REQUIRE(torch::allclose(A, B));
    REQUIRE(A.axis(0) == B.axis(0));
    REQUIRE(A.axis(1) == B.axis(1));
  }

  SECTION("tensor information")
  {
    auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
    REQUIRE(A.scalar_type() == default_dtype());
    REQUIRE(A.device() == default_device());
    REQUIRE(A.dim() == 3);
    REQUIRE(A.sizes() == TensorShape{nbatch, 8, 7});
    REQUIRE(A.size(0) == nbatch);
    REQUIRE(A.size(1) == 8);
    REQUIRE(A.size(2) == 7);
    REQUIRE(A.batched());
    REQUIRE(A.batch_dim() == 1);
    REQUIRE(A.batch_sizes() == TensorShape{nbatch});
    REQUIRE(A.batch_size(0) == nbatch);
    REQUIRE(A.base_dim() == 2);
    REQUIRE(A.base_sizes() == TensorShape{8, 7});
    REQUIRE(A.base_size(0) == 8);
    REQUIRE(A.base_size(1) == 7);
    REQUIRE(A.base_storage() == 56);
  }

  SECTION("batch_index")
  {
    SECTION("1D LabeledTensor")
    {
      auto A = LabeledVector::zeros(nbatch, {&info1});
      REQUIRE(A.batch_index({0}).sizes() == TensorShapeRef({8}));
    }

    SECTION("2D LabeledTensor")
    {
      auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
      REQUIRE(A.batch_index({0}).sizes() == TensorShapeRef({8, 7}));
    }
  }

  SECTION("base_index")
  {
    SECTION("1D LabeledTensor")
    {
      auto A = LabeledVector::zeros(nbatch, {&info1});
      REQUIRE(A.base_index({"first"}).sizes() == TensorShapeRef({nbatch, 6}));
      REQUIRE(A.base_index({"second"}).sizes() == TensorShapeRef({nbatch, 1}));
      REQUIRE(A.base_index({"third"}).sizes() == TensorShapeRef({nbatch, 1}));
    }

    SECTION("2D LabeledTensor")
    {
      auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
      REQUIRE(A.base_index({"first", "first"}).sizes() == TensorShapeRef({nbatch, 6, 1}));
      REQUIRE(A.base_index({"first", "second"}).sizes() == TensorShapeRef({nbatch, 6, 6}));
      REQUIRE(A.base_index({"second", "first"}).sizes() == TensorShapeRef({nbatch, 1, 1}));
      REQUIRE(A.base_index({"second", "second"}).sizes() == TensorShapeRef({nbatch, 1, 6}));
      REQUIRE(A.base_index({"third", "first"}).sizes() == TensorShapeRef({nbatch, 1, 1}));
      REQUIRE(A.base_index({"third", "second"}).sizes() == TensorShapeRef({nbatch, 1, 6}));
    }
  }

  SECTION("reinterpret")
  {
    SECTION("1D LabeledTensor")
    {
      auto A = LabeledVector::zeros(nbatch, {&info1});
      REQUIRE(A.reinterpret<SR2>({"first"}).sizes() == TensorShapeRef({nbatch, 6}));
      REQUIRE(A.reinterpret<Scalar>({"second"}).sizes() == TensorShapeRef({nbatch}));
      REQUIRE(A.reinterpret<Scalar>({"third"}).sizes() == TensorShapeRef({nbatch}));
    }

    SECTION("2D LabeledTensor")
    {
      auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
      REQUIRE(A.reinterpret<SR2>({"first", "first"}).sizes() == TensorShapeRef({nbatch, 6}));
      REQUIRE(A.reinterpret<Scalar>({"second", "first"}).sizes() == TensorShapeRef({nbatch}));
      REQUIRE(A.reinterpret<SR2>({"second", "second"}).sizes() == TensorShapeRef({nbatch, 6}));
      REQUIRE(A.reinterpret<Scalar>({"third", "first"}).sizes() == TensorShapeRef({nbatch}));
      REQUIRE(A.reinterpret<SR2>({"third", "second"}).sizes() == TensorShapeRef({nbatch, 6}));
    }
  }

  SECTION("base_index_put_")
  {
    SECTION("logically 1D LabeledTensor")
    {
      auto A = LabeledVector::zeros(nbatch, {&info1});
      A.base_index_put_({"first"}, Tensor::ones(6).batch_expand(nbatch));
      REQUIRE(torch::sum(A.base_index({"first"})).item<double>() == Catch::Approx(nbatch * 6));
      REQUIRE(torch::sum(A.base_index({"second"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"third"})).item<double>() == Catch::Approx(0));
    }

    SECTION("logically 2D LabeledTensor")
    {
      auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
      A.base_index_put_({"third", "second"}, Tensor::ones({1, 6}).batch_expand(nbatch));
      REQUIRE(torch::sum(A.base_index({"first", "first"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"first", "second"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"second", "first"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"second", "second"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"third", "first"})).item<double>() == Catch::Approx(0));
      REQUIRE(torch::sum(A.base_index({"third", "second"})).item<double>() ==
              Catch::Approx(nbatch * 6));
    }
  }

  SECTION("clone")
  {
    auto A = LabeledMatrix::zeros(nbatch, {&info1, &info2});
    auto B = A.clone();

    REQUIRE(A.axis(0) == B.axis(0));
    REQUIRE(A.axis(1) == B.axis(1));
    REQUIRE(torch::allclose(A.tensor(), B.tensor()));

    // Since B is a deep copy, modifying B shouldn't affect A.
    B.base_index_put_({"third", "second"}, Tensor::ones({1, 6}).batch_expand(nbatch));
    REQUIRE(torch::sum(A.base_index({"third", "second"})).item<double>() == Catch::Approx(0));
    REQUIRE(torch::sum(B.base_index({"third", "second"})).item<double>() ==
            Catch::Approx(nbatch * 6));
  }
}
