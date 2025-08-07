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

#include <ATen/Context.h>
#include <catch2/catch_test_macros.hpp>

#include "neml2/tensors/tensors.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("R3", "[tensors]")
{
  at::manual_seed(42);

  SECTION("levi_civita")
  {
    auto lc = R3::levi_civita();

    std::vector<TensorShape> p = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    std::vector<TensorShape> n = {{2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
    for (Size i = 0; i < 3; i++)
      for (Size j = 0; j < 3; j++)
        for (Size k = 0; k < 3; k++)
        {
          double v = 0.0;
          if (std::find(p.begin(), p.end(), TensorShape{i, j, k}) != p.end())
            v = 1.0;
          else if (std::find(n.begin(), n.end(), TensorShape{i, j, k}) != n.end())
            v = -1.0;
          REQUIRE_THAT(lc(i, j, k), test::allclose(Scalar(v, default_tensor_options())));
        }
  }
}
