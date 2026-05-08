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

#include "neml2/misc/defaults.h"
#include "neml2/tensors/TensorCache.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("TensorCache", "[tensors]")
{
  SECTION("cache miss invokes creator")
  {
    int call_count = 0;
    TensorCache cache(
        [&call_count](const TensorOptions & opts) -> Tensor
        {
          ++call_count;
          return Tensor::full({}, 3.14, opts);
        });

    const auto & t = cache(default_tensor_options());
    REQUIRE(call_count == 1);
    REQUIRE(t.item<double>() == Catch::Approx(3.14));
  }

  SECTION("second lookup with same options returns cached tensor (no extra call)")
  {
    int call_count = 0;
    TensorCache cache(
        [&call_count](const TensorOptions & opts) -> Tensor
        {
          ++call_count;
          return Tensor::full({}, 1.0, opts);
        });

    cache(default_tensor_options());
    cache(default_tensor_options());
    REQUIRE(call_count == 1);
  }

  SECTION("same tensor object is returned on repeated lookup")
  {
    TensorCache cache([](const TensorOptions & opts) { return Tensor::full({}, 7.0, opts); });

    const auto & first = cache(default_tensor_options());
    const auto & second = cache(default_tensor_options());
    REQUIRE(&first == &second);
  }

  SECTION("different dtype triggers a new creation")
  {
    int call_count = 0;
    TensorCache cache(
        [&call_count](const TensorOptions & opts) -> Tensor
        {
          ++call_count;
          return Tensor::full({}, 2.0, opts);
        });

    const auto opts64 = default_tensor_options().dtype(kFloat64);
    const auto opts32 = default_tensor_options().dtype(kFloat32);
    cache(opts64);
    cache(opts32);
    REQUIRE(call_count == 2);
  }

  SECTION("cached tensors respect the requested dtype")
  {
    TensorCache cache([](const TensorOptions & opts) -> Tensor
                      { return Tensor::full({}, 5.0, opts); });

    const auto t64 = cache(default_tensor_options().dtype(kFloat64));
    const auto t32 = cache(default_tensor_options().dtype(kFloat32));
    REQUIRE(t64.scalar_type() == kFloat64);
    REQUIRE(t32.scalar_type() == kFloat32);
  }
}
