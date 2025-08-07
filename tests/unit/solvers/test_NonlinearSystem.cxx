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
#include <catch2/matchers/catch_matchers_all.hpp>

#include "SampleNonlinearSystems.h"

using namespace neml2;

TEST_CASE("NonlinearSystem", "[solvers]")
{
  // Initial guess
  TensorShape batch_sz = {2};
  Size nbase = 4;
  auto x0 =
      NonlinearSystem::Sol<false>(Tensor::full(batch_sz, {}, nbase, 2.0, default_tensor_options()));

  // Create the nonlinear system
  auto options = PowerTestSystem::expected_options();
  options.set<bool>("automatic_scaling") = true;
  PowerTestSystem system(options);

  SECTION("Automatic scaling can reduce condition number")
  {
    system.init_scaling(x0);
    auto x0p = system.scale(x0);
    REQUIRE(at::max(at::linalg_cond(system.Jacobian(x0p))).item<double>() == Catch::Approx(1.0));
  }
}
