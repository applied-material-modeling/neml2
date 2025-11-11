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
#include <catch2/catch_template_test_macros.hpp>

#include "neml2/tensors/functions/vol.h"
#include "neml2/tensors/functions/tr.h"
#include "neml2/tensors/tensors.h"
#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("vol", "[tensors/functions]")
{
  at::manual_seed(42);
  auto cfg = test::generate_tensor_config();

  SECTION("R2")
  {
    auto shape = test::generate_tensor_shape<R2>();
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
      auto A = test::generate_random_tensor<R2>(cfg, shape);
      auto v = neml2::vol(A);

      REQUIRE(test::match_tensor_shape(v, shape));

      auto ref = neml2::tr(A) / 3.0 * R2::identity(A.options());
      REQUIRE_THAT(v, test::allclose(ref));
    }
  }

  SECTION("SR2")
  {
    auto shape = test::generate_tensor_shape<SR2>();
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
      auto A = test::generate_random_tensor<SR2>(cfg, shape);
      auto v = neml2::vol(A);

      REQUIRE(test::match_tensor_shape(v, shape));

      auto ref = neml2::tr(A) / 3.0 * SR2::identity(A.options());
      REQUIRE_THAT(v, test::allclose(ref));
    }
  }

  SECTION("WR2")
  {
    auto shape = test::generate_tensor_shape<WR2>();
    DYNAMIC_SECTION(cfg.desc() << " shape: " << shape.desc())
    {
      auto A = test::generate_random_tensor<WR2>(cfg, shape);
      auto v = neml2::vol(A);

      REQUIRE(test::match_tensor_shape(v, shape));

      auto ref = WR2::zeros(A.options()).batch_expand_as(A);
      REQUIRE_THAT(v, test::allclose(ref));
    }
  }
}
