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
#include "neml2/tensors/functions/abs.h"
#include "neml2/tensors/functions/linalg/eigh.h"

using namespace neml2;
using namespace indexing;

TEST_CASE("eigh", "[tensors/functions/linalg]")
{
  SECTION("eigh function")
  {
    auto s = SR2::fill(-0.3482, 0.3482, 0, 0.087045, 0.087045, 0.78333);
    auto ss = s.dynamic_expand({5, 4, 1, 2});
    auto b = Vec::fill(-0.858002364, -0.0158323254, 0.8738346695);
    auto v = R2::fill(0.83927690982819,
                      -0.04919575527310,
                      0.54147386550903,
                      -0.54287183284760,
                      -0.13090363144875,
                      0.82955050468445,
                      -0.03007052093744,
                      0.99017369747162,
                      0.13657139241695);
    auto [eigvals, eigvecs] = linalg::eigh(s);
    REQUIRE(at::allclose(eigvals, b));
    for (Size i = 0; i < 3; i++)
    {
      auto v1 = eigvecs.base_index({Slice(), i});
      auto v2 = v.base_index({Slice(), i});
      REQUIRE((at::allclose(v1, v2) || at::allclose(v1, -v2)));
    }
  }
}
