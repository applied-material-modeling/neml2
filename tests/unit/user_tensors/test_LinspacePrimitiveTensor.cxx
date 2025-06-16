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
#include "neml2/tensors/tensors.h"

using namespace neml2;

#define test_LinspacePrimitiveTensor(tensor_type, tensor_name, batch_shape, nstep, dim)            \
  SECTION("Linspace" #tensor_type)                                                                 \
  {                                                                                                \
    const auto tensor_name = factory->get_object<tensor_type>("Tensors", #tensor_name);            \
    const auto tensor_name##_start =                                                               \
        factory->get_object<tensor_type>("Tensors", #tensor_name "0");                             \
    const auto tensor_name##_end = factory->get_object<tensor_type>("Tensors", #tensor_name "1");  \
    const auto tensor_name##_correct =                                                             \
        tensor_type::linspace(*tensor_name##_start, *tensor_name##_end, nstep, dim);               \
    REQUIRE(tensor_name->batch_sizes() == batch_shape);                                            \
    REQUIRE(tensor_name->base_sizes() == tensor_type::const_base_sizes);                           \
    REQUIRE(at::allclose(*tensor_name, tensor_name##_correct));                                    \
  }                                                                                                \
  static_assert(true)

TEST_CASE("LinspacePrimitiveTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_LinspacePrimitiveTensor.i");

  TensorShape B{100, 2, 1};
  Size nstep = 100;
  Size dim = 0;

  test_LinspacePrimitiveTensor(Scalar, a, B, nstep, dim);
  test_LinspacePrimitiveTensor(Vec, b, B, nstep, dim);
  test_LinspacePrimitiveTensor(Rot, c, B, nstep, dim);
  test_LinspacePrimitiveTensor(R2, d, B, nstep, dim);
  test_LinspacePrimitiveTensor(SR2, e, B, nstep, dim);
  test_LinspacePrimitiveTensor(R3, f, B, nstep, dim);
  test_LinspacePrimitiveTensor(SFR3, g, B, nstep, dim);
  test_LinspacePrimitiveTensor(R4, h, B, nstep, dim);
  test_LinspacePrimitiveTensor(SFR4, i, B, nstep, dim);
  test_LinspacePrimitiveTensor(WFR4, j, B, nstep, dim);
  test_LinspacePrimitiveTensor(SSR4, k, B, nstep, dim);
  test_LinspacePrimitiveTensor(R5, l, B, nstep, dim);
  test_LinspacePrimitiveTensor(SSFR5, m, B, nstep, dim);
}
