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

#include "neml2/neml2.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/exp.h"

using namespace neml2;

TEST_CASE("GaussianTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_GaussianTensor.i");

  SECTION("load correctly")
  {
    const auto g = factory->get_object<Scalar>("Tensors", "g");
    REQUIRE(g->dynamic_sizes() == TensorShape{3});
    REQUIRE(g->intmd_sizes() == TensorShape{});
    REQUIRE(g->base_sizes() == TensorShape{});

    const auto points = factory->get_object<Scalar>("Tensors", "points");
    const auto width = factory->get_object<Scalar>("Tensors", "width");
    const auto height = factory->get_object<Scalar>("Tensors", "height");
    const auto center = factory->get_object<Scalar>("Tensors", "center");

    const auto z = (*points - *center) / *width;
    const auto expected = *height * neml2::exp(-0.5 * z * z);

    REQUIRE(at::allclose(*g, expected));
  }
}
