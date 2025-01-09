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

#include "neml2/dispatchers/ValueMapLoader.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"

using namespace neml2;

TEST_CASE("ValueMapLoader", "[dispatchers]")
{
  const auto strain_name = VariableName{"state", "strain"};
  const auto strain0 = SR2::fill(0.1, 0.05, -0.01).batch_expand({5, 5});
  const auto strain1 = SR2::fill(0.2, 0.1, 0).batch_expand({5, 5});
  const auto strain = SR2::linspace(strain0, strain1, 100, 1);
  const auto temperature_name = VariableName{"forces", "temperature"};
  const auto temperature = Scalar::full(300).batch_expand({5, 1, 5});
  const auto value_map = ValueMap{{strain_name, strain}, {temperature_name, temperature}};

  ValueMapLoader loader(value_map, 1);
  REQUIRE(loader.total() == 100);
  REQUIRE(loader.offset() == 0);

  std::size_t n;
  ValueMap work;

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(1);
  REQUIRE(loader.offset() == 1);
  REQUIRE(n == 1);
  REQUIRE(work[strain_name].batch_sizes() == TensorShape{5, 1, 5});
  REQUIRE(work[strain_name].base_sizes() == TensorShape{6});
  REQUIRE(at::allclose(work[strain_name], strain.slice(1, 0, 1)));
  REQUIRE(work[temperature_name].batch_sizes() == TensorShape{5, 1, 5});
  REQUIRE(work[temperature_name].base_sizes() == TensorShape{});
  REQUIRE(at::allclose(work[temperature_name], temperature.slice(1, 0, 1)));

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(2);
  REQUIRE(loader.offset() == 3);
  REQUIRE(n == 2);
  REQUIRE(work[strain_name].batch_sizes() == TensorShape{5, 2, 5});
  REQUIRE(work[strain_name].base_sizes() == TensorShape{6});
  REQUIRE(at::allclose(work[strain_name], strain.slice(1, 1, 3)));
  REQUIRE(work[temperature_name].batch_sizes() == TensorShape{5, 1, 5});
  REQUIRE(work[temperature_name].base_sizes() == TensorShape{});
  REQUIRE(at::allclose(work[temperature_name], temperature.slice(1, 0, 1)));

  REQUIRE(loader.has_more());
  std::tie(n, work) = loader.next(1000);
  REQUIRE(loader.offset() == 100);
  REQUIRE(n == 97);
  REQUIRE(work[strain_name].batch_sizes() == TensorShape{5, 97, 5});
  REQUIRE(work[strain_name].base_sizes() == TensorShape{6});
  REQUIRE(at::allclose(work[strain_name], strain.slice(1, 3, 100)));
  REQUIRE(work[temperature_name].batch_sizes() == TensorShape{5, 1, 5});
  REQUIRE(work[temperature_name].base_sizes() == TensorShape{});
  REQUIRE(at::allclose(work[temperature_name], temperature.slice(1, 0, 1)));

  REQUIRE(!loader.has_more());
}
