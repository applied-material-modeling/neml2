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

#include "neml2/tensors/tensors.h"
#include "neml2/tensors/crystallography.h"

#include "neml2/tensors/functions/det.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("Misorientation calculations", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  SECTION("misorientation")
  {
    SECTION("shapes")
    {
      SECTION("simple dynamic")
      {
        auto r1 = Rot::rand({10}, {}, DTO);
        auto r2 = Rot::rand({10}, {}, DTO);
        auto miso = crystallography::misorientation(r1, r2, "432");
        REQUIRE(miso.sizes() == at::IntArrayRef({10}));
      }
      SECTION("broadcast dynamic")
      {
        auto r1 = Rot::rand({10, 1}, {}, DTO);
        auto r2 = Rot::rand({15}, {}, DTO);
        auto miso = crystallography::misorientation(r1, r2, "432");
        REQUIRE(miso.sizes() == at::IntArrayRef({10, 15}));
      }
      SECTION("existing intermediate")
      {
        auto r1 = Rot::rand({10, 1}, {2}, DTO);
        auto r2 = Rot::rand({15}, {2}, DTO);
        auto miso = crystallography::misorientation(r1, r2, "432");
        REQUIRE(miso.sizes() == at::IntArrayRef({10, 15, 2}));
      }
    }
    SECTION("correctness")
    {
      SECTION("same")
      {
        auto r1 = Rot::rand({10}, {}, DTO);
        auto miso = crystallography::misorientation(r1, r1, "432");
        REQUIRE_THAT(miso, test::allclose(Scalar::zeros_like(miso), 1e-6, 1e-6));
      }
      SECTION("max 432")
      {
        auto r1 = Rot::fill_euler_angles(Vec::fill(0, 0, 0, DTO), "bunge", "degrees");
        auto r2 = Rot::fill_euler_angles(Vec::fill(45, 45, 0, DTO), "bunge", "degrees");
        auto miso = crystallography::misorientation(r1, r2, "432");
        REQUIRE_THAT(miso, test::allclose(Scalar::full_like(miso, 1.09605), 1e-3, 1e-6));
      }
      SECTION("one axis")
      {
        auto r1 = Rot::fill_euler_angles(Vec::fill(0, 0, 0, DTO), "bunge", "degrees");
        auto r2 = Rot::fill_euler_angles(Vec::fill(10, 0, 0, DTO), "bunge", "degrees");
        auto miso = crystallography::misorientation(r1, r2, "432");
        REQUIRE_THAT(miso, test::allclose(Scalar::full_like(miso, 0.174532925), 1e-6, 1e-6));
      }
      SECTION("symmetry")
      {
        auto r1 = Rot::rand({10}, {}, DTO);
        auto r2 = Rot::rand({10}, {}, DTO);
        auto a = crystallography::misorientation(r1, r2, "432");
        auto b = crystallography::misorientation(r2, r1, "432");
        REQUIRE_THAT(a, test::allclose(b));
      }
      SECTION("invariance")
      {
        auto r1 = Rot::rand({}, {}, DTO);
        auto r2 = Rot::rand({}, {}, DTO);
        for (auto orbifold : {"1", "2", "222", "4", "42", "3", "6", "32", "622", "23", "432"})
        {
          auto a = crystallography::misorientation(r1, r2, orbifold);
          auto ops = crystallography::symmetry(orbifold, DTO);
          for (Size i = 0; i < ops.intmd_size(0); i++)
          {
            auto Ri = ops.intmd_index({i});
            for (Size j = 0; j < ops.intmd_size(0); j++)
            {
              auto Rj = ops.intmd_index({j});
              auto r1p = Rot::fill_matrix(Ri * r1.euler_rodrigues());
              auto r2p = Rot::fill_matrix(Rj * r2.euler_rodrigues());

              auto b = crystallography::misorientation(r1p, r2p, orbifold);
              REQUIRE_THAT(a, test::allclose(b, 1e-6, 1e-6));
              auto c = crystallography::misorientation(r1, r2p, orbifold);
              REQUIRE_THAT(a, test::allclose(c, 1e-6, 1e-6));
              auto d = crystallography::misorientation(r1p, r2, orbifold);
              REQUIRE_THAT(a, test::allclose(d, 1e-6, 1e-6));
            }
          }
        }
      }
    }
  }
}
