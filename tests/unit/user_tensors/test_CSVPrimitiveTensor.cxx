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

#include "neml2/base/Factory.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

#ifdef NEML2_HAS_CSV

TEST_CASE("CSVPrimitiveTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_CSVPrimitiveTensor.i");
  const auto tensor_size = TensorShape{4};

  SECTION("Parse CSV correctly")
  {
    const auto a = factory->get_object<Scalar>("Tensors", "a");
    REQUIRE(at::allclose(*a, Scalar::create({0, 1, 2, 3})));
    REQUIRE(a->batch_sizes() == tensor_size);
    REQUIRE(a->base_sizes() == Scalar::const_base_sizes);

    const auto b = factory->get_object<Vec>("Tensors", "b");
    REQUIRE(at::allclose(*b, Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}})));
    REQUIRE(b->batch_sizes() == tensor_size);
    REQUIRE(b->base_sizes() == Vec::const_base_sizes);

    const auto c = factory->get_object<SR2>("Tensors", "c");
    REQUIRE(at::allclose(
        *c,
        SR2::create(
            {{2, 3, 4, 5, 6, 7}, {3, 4, 5, 6, 7, 8}, {4, 5, 6, 7, 8, 9}, {5, 6, 7, 8, 9, 10}})));
    REQUIRE(c->batch_sizes() == tensor_size);
    REQUIRE(c->base_sizes() == SR2::const_base_sizes);
  }
}

#endif