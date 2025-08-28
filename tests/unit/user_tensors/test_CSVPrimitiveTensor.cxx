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

TEST_CASE("CSVPrimitiveTensor", "[user_tensors]")
{
  SECTION("Read and reshape CSV  correctly")
  {
    auto factory = load_input("user_tensors/test_CSVPrimitiveTensor.i");

    const auto a = factory->get_object<Scalar>("Tensors", "a");
    const auto a_correct = Scalar::create({5, 6, 7, 8});
    REQUIRE(at::allclose(*a, a_correct));
    REQUIRE(a->batch_sizes() == TensorShape{4});
    REQUIRE(a->base_sizes() == Scalar::const_base_sizes);

    const auto b = factory->get_object<Vec>("Tensors", "b");
    const auto b_correct =
        Vec::create({{0.0, 0.5, 0.4}, {0.1, -0.1, 0.0}, {0.2, 0.0, 0.2}, {0.3, 0.2, -0.1}});
    REQUIRE(at::allclose(*b, b_correct));
    REQUIRE(b->batch_sizes() == TensorShape{4});
    REQUIRE(b->base_sizes() == Vec::const_base_sizes);

    // const auto c = factory->get_object<Vec>("Tensors", "c");
    // const auto c_correct =
    //     Vec::create({{{0.0, 0.5, 0.4}, {0.1, -0.1, 0.0}}, {{0.2, 0.0, 0.2}, {0.3, 0.2, -0.1}}});
    // REQUIRE(at::allclose(*c, c_correct));
    // REQUIRE(c->batch_sizes() == TensorShape{2, 2});
    // REQUIRE(c->base_sizes() == Vec::const_base_sizes);

    // const auto d = factory->get_object<SR2>("Tensors", "d");
    // const auto d_correct = SR2::create({{0.5, 1.1, 2.2, 0.5, 1.1, 2.2},
    //                                     {-1.5, 2.2, 3.3, 0.5, 1.1, 2.2},
    //                                     {-3.0, -5.0, 100.1, 0.5, 1.1, 2.2},
    //                                     {20.1, -1.1, -13.0, 0.5, 1.1, 2.2}});
    // REQUIRE(at::allclose(*d, d_correct));
    // REQUIRE(d->batch_sizes() == TensorShape{4});
    // REQUIRE(d->base_sizes() == SR2::const_base_sizes);
  }
}