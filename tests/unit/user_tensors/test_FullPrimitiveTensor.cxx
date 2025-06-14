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

#include "neml2/tensors/tensors.h"

using namespace neml2;

#define test_FullPrimitiveTensor(tensor_type, tensor_name, batch_shape, value)                     \
  SECTION("Full" #tensor_type)                                                                     \
  {                                                                                                \
    const auto tensor_name = factory.get_object<tensor_type>("Tensors", #tensor_name);             \
    REQUIRE(tensor_name->batch_sizes() == batch_shape);                                            \
    REQUIRE(tensor_name->base_sizes() == tensor_type::const_base_sizes);                           \
    REQUIRE(at::allclose(*tensor_name,                                                             \
                         tensor_type::full(batch_shape, value, default_tensor_options())));        \
  }                                                                                                \
  static_assert(true)

TEST_CASE("FullPrimitiveTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_FullPrimitiveTensor.i");

  TensorShape B{2, 1};

  test_FullPrimitiveTensor(Scalar, a, B, 1.3);
  test_FullPrimitiveTensor(Vec, b, B, 1.3);
  test_FullPrimitiveTensor(Rot, c, B, 1.3);
  test_FullPrimitiveTensor(R2, d, B, 1.3);
  test_FullPrimitiveTensor(SR2, e, B, 1.3);
  test_FullPrimitiveTensor(R3, f, B, 1.3);
  test_FullPrimitiveTensor(SFR3, g, B, 1.3);
  test_FullPrimitiveTensor(R4, h, B, 1.3);
  test_FullPrimitiveTensor(SFR4, i, B, 1.3);
  test_FullPrimitiveTensor(WFR4, j, B, 1.3);
  test_FullPrimitiveTensor(SSR4, k, B, 1.3);
  test_FullPrimitiveTensor(R5, l, B, 1.3);
  test_FullPrimitiveTensor(SSFR5, m, B, 1.3);
}
