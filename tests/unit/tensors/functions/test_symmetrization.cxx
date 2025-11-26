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

#include "neml2/tensors/functions/symmetrization.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("symmetrization", "[tensors/functions]")
{
  SECTION("mandel_factor")
  {
    REQUIRE(mandel_factor(0) == 1.0);
    REQUIRE(mandel_factor(1) == 1.0);
    REQUIRE(mandel_factor(2) == 1.0);
    REQUIRE(mandel_factor(3) == sqrt2);
    REQUIRE(mandel_factor(4) == sqrt2);
    REQUIRE(mandel_factor(5) == sqrt2);

#ifndef NDEBUG
    REQUIRE_THROWS(mandel_factor(-1));
    REQUIRE_THROWS(mandel_factor(6));
#endif
  }

  SECTION("full_to_mandel mandel_to_full")
  {
    auto full = R2::rand({2, 3}, {3});
    auto full_sym = 0.5 * (full + full.transpose());
    auto mandel = full_to_mandel(full_sym, 0);
    auto mandel_expected = base_stack({full_sym(0, 0),
                                       full_sym(1, 1),
                                       full_sym(2, 2),
                                       sqrt2 * full_sym(1, 2),
                                       sqrt2 * full_sym(0, 2),
                                       sqrt2 * full_sym(0, 1)});
    REQUIRE_THAT(mandel, test::allclose(mandel_expected));

    // make a round trip
    auto full_reconstructed = mandel_to_full(mandel, 0);
    REQUIRE_THAT(full_reconstructed, test::allclose(full_sym));
  }

  SECTION("full_to_skew skew_to_full")
  {
    auto full = R2::rand({2, 3}, {3});
    auto full_skew = 0.5 * (full - full.transpose());
    auto skew = full_to_skew(full_skew, 0);
    auto skew_expected = base_stack({full_skew(2, 1), full_skew(0, 2), full_skew(1, 0)});
    REQUIRE_THAT(skew, test::allclose(skew_expected));

    // make a round trip
    auto full_reconstructed = skew_to_full(skew, 0);
    REQUIRE_THAT(full_reconstructed, test::allclose(full_skew));
  }
}
