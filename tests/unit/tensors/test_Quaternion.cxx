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

#include "neml2/tensors/Quaternion.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Rot.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("Quaternion", "[tensors]")
{
  SECTION("constructors")
  {
    SECTION("Rot")
    {
      auto r = Rot::fill(0.1, 0.2, 0.3);
      auto q = Quaternion(r);
      auto q_expected = Quaternion::fill(0.75438596491, 0.1754386, 0.35087719, 0.52631579);
      REQUIRE_THAT(q, test::allclose(q_expected));
    }
  }

  SECTION("rotation_matrix")
  {
    auto q = Quaternion::fill(-0.30411437, -0.15205718, 0.91234311, 0.22808578);
    auto R2 = R2::fill(-0.76878613,
                       -0.13872832,
                       -0.62427746,
                       -0.41618497,
                       0.84971098,
                       0.32369942,
                       0.48554913,
                       0.50867052,
                       -0.71098266);
    REQUIRE_THAT(q.rotation_matrix(), test::allclose(R2));
  }

  SECTION("dist")
  {
    auto q1 = Quaternion::fill(0.5, 0.2, 0.2, 0.5);
    auto q2 = Quaternion::fill(0.6, 0.5, 0.5, 0.4);
    auto q3 = Quaternion::fill(-0.8, -0.9, -0.8, -0.9);
    auto dist12 = Scalar(2 * std::acos(0.7), default_tensor_options());
    auto dist13 = Scalar(2 * std::acos(1.0), default_tensor_options());
    REQUIRE_THAT(q1.dist(q2), test::allclose(dist12));
    REQUIRE_THAT(q1.dist(q3), test::allclose(dist13));
  }
}
