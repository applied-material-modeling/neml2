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
#include "neml2/dispatchers/valuemap_helpers.h"
#include "neml2/dispatchers/StaticHybridScheduler.h"
#include "neml2/dispatchers/WorkDispatcher.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/functions/linspace.h"

using namespace neml2;

TEST_CASE("WorkDispatcher ValueMapLoader StaticHybridScheduler", "[dispatchers]")
{
  if (!torch::cuda::is_available())
    SKIP("cuda not available");

  // Along which batch dimension to dispatch work
  const Size dynamic_dim = 1;

  const auto strain_name = VariableName{"state", "strain"};
  const auto strain0 = SR2::fill(0.1, 0.05, -0.01).dynamic_expand({5, 5});
  const auto strain1 = SR2::fill(0.2, 0.1, 0).dynamic_expand({5, 5});
  const auto strain = dynamic_linspace(strain0, strain1, 100, dynamic_dim);

  const auto temperature_name = VariableName{"forces", "temperature"};
  const auto temperature = Scalar::full(300).dynamic_expand({5, 1, 5});
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
  { return valuemap_cat_reduce(std::move(results), dynamic_dim); };

  auto post = [](ValueMap && x) -> ValueMap
  { return valuemap_move_device(std::move(x), Device("cpu")); };

  ValueMapLoader loader(x, dynamic_dim);

  OptionSet options = StaticHybridScheduler::expected_options();
  options.set<std::vector<Device>>("devices") = {Device("cuda:0"), Device("cpu")};
  options.set<std::vector<std::size_t>>("batch_sizes") = {23, 17};
  options.set<std::vector<std::size_t>>("capacities") = {55, 18};
  options.set<std::vector<double>>("priorities") = {1, 2};

  StaticHybridScheduler scheduler(options);
  scheduler.setup();

  SECTION("run_sync")
  {
    WorkDispatcher</*I=*/ValueMap,
                   /*O=*/ValueMap,
                   /*Of=*/ValueMap,
                   /*Ip=*/ValueMap,
                   /*Op=*/ValueMap>
        dispatcher(scheduler, false, func, red, &valuemap_move_device, post);
    auto y = dispatcher.run(loader);
    REQUIRE(at::allclose(y[stress_name], stress));
  }

  SECTION("run_async")
  {
    WorkDispatcher</*I=*/ValueMap,
                   /*O=*/ValueMap,
                   /*Of=*/ValueMap,
                   /*Ip=*/ValueMap,
                   /*Op=*/ValueMap>
        dispatcher(scheduler, true, func, red, &valuemap_move_device, post);
    auto y = dispatcher.run(loader);
    REQUIRE(at::allclose(y[stress_name], stress));
  }
}
