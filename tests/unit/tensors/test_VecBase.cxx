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

#include "utils.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("VecBase", "[tensors]")
{
  SECTION("rotate")
  {
    auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991);
    auto v = Vec::fill(1.0, -2.0, 3.0);
    auto vp = Vec::fill(0.495655, 3.13461, 1.98205);
    REQUIRE(at::allclose(v.rotate(r), vp));

    auto apply = [v](const Tensor & x) { return v.rotate(Rot(x)); };
    auto dvp_dr = finite_differencing_derivative(apply, r);
    REQUIRE(at::allclose(v.drotate(r), dvp_dr, 1e-4));

    auto R = r.euler_rodrigues();
    auto apply_R = [v](const Tensor & x) { return v.rotate(R2(x)); };
    auto dvp_dR = finite_differencing_derivative(apply_R, R);
    REQUIRE(at::allclose(v.drotate(R), dvp_dR, 1e-4));
  }
}
