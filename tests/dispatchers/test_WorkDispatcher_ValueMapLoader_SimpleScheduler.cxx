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

#include <torch/cuda.h>

#include "neml2/dispatchers/ValueMapLoader.h"
#include "neml2/dispatchers/SimpleScheduler.h"
#include "neml2/dispatchers/WorkDispatcher.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/functions/cat.h"

using namespace neml2;

TEST_CASE("WorkDispatcher ValueMapLoader SimpleScheduler", "[dispatchers]")
{
  // Along which batch dimension to dispatch work
  const Size batch_dim = 1;

  const auto strain_name = VariableName{"state", "strain"};
  const auto strain0 = SR2::fill(0.1, 0.05, -0.01).batch_expand({5, 5});
  const auto strain1 = SR2::fill(0.2, 0.1, 0).batch_expand({5, 5});
  const auto strain = SR2::linspace(strain0, strain1, 100, batch_dim);

  const auto temperature_name = VariableName{"forces", "temperature"};
  const auto temperature = Scalar::full(300).batch_expand({5, 1, 5});
  const auto x = ValueMap{{strain_name, strain}, {temperature_name, temperature}};

  const auto stress_name = VariableName{"state", "stress"};
  const auto stress = strain * temperature; // Not a "stress" but...

  auto func = [&strain_name, &temperature_name, &stress_name](ValueMap && x,
                                                              Device /*device*/) -> ValueMap
  {
    const auto & strain = x[strain_name];
    const auto & temperature = x[temperature_name];
    return ValueMap{{stress_name, strain * Scalar(temperature)}};
  };
  auto red = [](std::vector<ValueMap> && results) -> ValueMap
  {
    // Re-bin the results
    std::map<VariableName, std::vector<Tensor>> vars;
    for (auto && result : results)
      for (auto && [name, value] : result)
        vars[name].emplace_back(std::move(value));

    // Concatenate the tensors
    ValueMap ret;
    for (auto && [name, values] : vars)
      ret[name] = batch_cat(values, batch_dim);

    return ret;
  };
  auto pre = [](ValueMap && x, Device device) -> ValueMap
  {
    // Move the tensors to the device
    for (auto && [name, value] : x)
      x[name] = value.to(device);
    return x;
  };
  auto post = [](ValueMap && x) -> ValueMap { return std::move(x); };

  ValueMapLoader loader(x, batch_dim);
  WorkDispatcher</*I=*/ValueMap, /*O=*/ValueMap, /*Of=*/ValueMap, /*Ip=*/ValueMap, /*Op=*/ValueMap>
      dispatcher(func, red, pre, post);

  SECTION("cpu")
  {
    SimpleScheduler scheduler(kCPU, 23, 55);

    SECTION("run")
    {
      auto y = dispatcher.run(loader, scheduler);
      REQUIRE(at::allclose(y[stress_name], stress));
    }

    SECTION("run_async")
    {
      auto y = dispatcher.run_async(loader, scheduler);
      REQUIRE(at::allclose(y[stress_name], stress));
    }
  }

  SECTION("cuda")
  {
    if (!torch::cuda::is_available())
      SKIP("cuda not available");

    auto device = Device("cuda:0");
    SimpleScheduler scheduler(device, 23, 55);

    SECTION("run")
    {
      auto y = dispatcher.run(loader, scheduler);
      REQUIRE(y[stress_name].device() == device);
      REQUIRE(at::allclose(y[stress_name].to(kCPU), stress));
    }

    SECTION("run_async")
    {
      auto y = dispatcher.run_async(loader, scheduler);
      REQUIRE(y[stress_name].device() == device);
      REQUIRE(at::allclose(y[stress_name].to(kCPU), stress));
    }
  }
}
