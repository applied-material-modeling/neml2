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

#include "neml2/dispatcher/UniformSequentialValueMapDispatcher.h"
#include "neml2/models/Model.h"

using namespace neml2;

TEST_CASE("UniformSequentialValueMapDispatcher", "[dispatcher]")
{
  const auto strain_name = VariableName{"state", "strain"};
  const auto strain0 = SR2::fill(0.1, 0.05, -0.01).batch_expand({5, 5});
  const auto strain1 = SR2::fill(0.2, 0.1, 0).batch_expand({5, 5});
  const auto strain = SR2::linspace(strain0, strain1, 100, 0);

  const auto temperature_name = VariableName{"forces", "temperature"};
  const auto temperature = Scalar::full(300).batch_expand({1, 5, 5});
  const auto x = ValueMap{{strain_name, strain}, {temperature_name, temperature}};

  const auto stress_name = VariableName{"state", "stress"};
  const auto stress = strain * temperature; // A dummy stress

  // A dummy function to test the dispatcher
  auto func = [&strain_name, &temperature_name, &stress_name](ValueMap && x) -> ValueMap
  {
    const auto & strain = x[strain_name];
    const auto & temperature = x[temperature_name];
    return ValueMap{{stress_name, strain * Scalar(temperature)}};
  };

  // Dispatch along 1st batch dimension with a batch size of 23
  UniformSequentialValueMapDispatcher dispatcher(1, 23, func);
  auto y = dispatcher(x);
  REQUIRE(torch::allclose(y[stress_name], stress));
}
