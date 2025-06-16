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
#include "neml2/tensors/SR2.h"

using namespace neml2;

TEST_CASE("FillSR2", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_FillSR2.i");

  const auto auto_1 = factory->get_object<SR2>("Tensors", "1");
  const auto auto_1_correct = SR2::fill(1);
  REQUIRE(at::allclose(*auto_1, auto_1_correct));

  const auto auto_3 = factory->get_object<SR2>("Tensors", "3");
  const auto auto_3_correct = SR2::fill(1, 2, 3);
  REQUIRE(at::allclose(*auto_3, auto_3_correct));

  const auto auto_6 = factory->get_object<SR2>("Tensors", "6");
  const auto auto_6_correct = SR2::fill(1, 2, 3, 4, 5, 6);
  REQUIRE(at::allclose(*auto_6, auto_6_correct));
}
