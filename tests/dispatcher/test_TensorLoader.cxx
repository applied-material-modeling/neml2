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

#include "neml2/dispatcher/TensorLoader.h"

using namespace neml2;

TEST_CASE("TensorLoader", "[dispatcher]")
{
  auto start = Tensor::zeros({5, 5}, {2, 3});
  auto end = Tensor::full({5, 5}, {2, 3}, 100.0);
  auto ten = Tensor::linspace(start, end, 100, 1);
  REQUIRE(ten.batch_sizes() == TensorShape{5, 100, 5});
  REQUIRE(ten.base_sizes() == TensorShape{2, 3});

  TensorLoader loader(ten, 1);
  REQUIRE(loader.total() == 100);
  REQUIRE(loader.offset() == 0);

  std::size_t n;
  Tensor work;

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(1);
  REQUIRE(loader.offset() == 1);
  REQUIRE(n == 1);
  REQUIRE(work.batch_sizes() == TensorShape{5, 1, 5});
  REQUIRE(work.base_sizes() == TensorShape{2, 3});
  REQUIRE(torch::allclose(work, ten.slice(1, 0, 1)));

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(2);
  REQUIRE(loader.offset() == 3);
  REQUIRE(n == 2);
  REQUIRE(work.batch_sizes() == TensorShape{5, 2, 5});
  REQUIRE(work.base_sizes() == TensorShape{2, 3});
  REQUIRE(torch::allclose(work, ten.slice(1, 1, 3)));

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(1000);
  REQUIRE(loader.offset() == 100);
  REQUIRE(n == 97);
  REQUIRE(work.batch_sizes() == TensorShape{5, 97, 5});
  REQUIRE(work.base_sizes() == TensorShape{2, 3});
  REQUIRE(torch::allclose(work, ten.slice(1, 3)));

  REQUIRE(!loader.has_more());
}
