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

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/R3.h"
#include "neml2/tensors/SFR3.h"
#include "neml2/tensors/R4.h"
#include "neml2/tensors/SFR4.h"
#include "neml2/tensors/WFR4.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/R5.h"
#include "neml2/tensors/SSFR5.h"

using namespace neml2;

TEST_CASE("PrimitiveTensor", "[tensors]")
{
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2}; // batch shape
  Size Bn = B.size();           // batch dimension

  SECTION("class PrimitiveTensor")
  {
    SECTION("const_base_sizes")
    {
      REQUIRE(Scalar::const_base_sizes == TensorShape{});
      REQUIRE(Vec::const_base_sizes == TensorShape{3});
      REQUIRE(R2::const_base_sizes == TensorShape{3, 3});
      REQUIRE(SR2::const_base_sizes == TensorShape{6});
      REQUIRE(R3::const_base_sizes == TensorShape{3, 3, 3});
      REQUIRE(SFR3::const_base_sizes == TensorShape{6, 3});
      REQUIRE(SFR4::const_base_sizes == TensorShape{6, 3, 3});
      REQUIRE(WFR4::const_base_sizes == TensorShape{3, 3, 3});
      REQUIRE(R4::const_base_sizes == TensorShape{3, 3, 3, 3});
      REQUIRE(SSR4::const_base_sizes == TensorShape{6, 6});
      REQUIRE(R5::const_base_sizes == TensorShape{3, 3, 3, 3, 3});
      REQUIRE(SSFR5::const_base_sizes == TensorShape{6, 6, 3});
    }

    SECTION("const_base_dim")
    {
      REQUIRE(Scalar::const_base_dim == 0);
      REQUIRE(Vec::const_base_dim == 1);
      REQUIRE(R2::const_base_dim == 2);
      REQUIRE(SR2::const_base_dim == 1);
      REQUIRE(R3::const_base_dim == 3);
      REQUIRE(SFR3::const_base_dim == 2);
      REQUIRE(R4::const_base_dim == 4);
      REQUIRE(SFR4::const_base_dim == 3);
      REQUIRE(WFR4::const_base_dim == 3);
      REQUIRE(SSR4::const_base_dim == 2);
      REQUIRE(R5::const_base_dim == 5);
      REQUIRE(SSFR5::const_base_dim == 3);
    }

    SECTION("PrimitiveTensor")
    {
      SECTION("default")
      {
        R5 a;
        REQUIRE(!a.defined());
      }

      SECTION("from Tensor and batch dimension")
      {
        auto a = at::full(utils::add_shapes(B, R5::const_base_sizes), 3.1231, DTO);
        auto b = R5(a, Bn);
        REQUIRE(b.batch_dim() == Bn);
        REQUIRE(b.base_dim() == R5::const_base_dim);
        REQUIRE(b.batch_sizes() == B);
        REQUIRE(b.base_sizes() == R5::const_base_sizes);
        REQUIRE(b.base_storage() == R5::const_base_storage);

#ifndef NDEBUG
        // Calling .defined() to make sure this doesn't get optimized away...
        REQUIRE_THROWS_WITH(SR2(a, Bn).defined(),
                            Catch::Matchers::ContainsSubstring("Base shape mismatch"));
#endif
      }

      SECTION("from Tensor")
      {
        auto a = at::full(utils::add_shapes(B, R5::const_base_sizes), 3.1231, DTO);
        auto b = R5(a);
        REQUIRE(b.batch_dim() == Bn);
        REQUIRE(b.base_dim() == R5::const_base_dim);
        REQUIRE(b.batch_sizes() == B);
        REQUIRE(b.base_sizes() == R5::const_base_sizes);
        REQUIRE(b.base_storage() == R5::const_base_storage);

#ifndef NDEBUG
        // Calling .defined() to make sure this doesn't get optimized away...
        REQUIRE_THROWS_WITH(SR2(a).defined(),
                            Catch::Matchers::ContainsSubstring("Base shape mismatch"));
#endif
      }
    }

    SECTION("empty")
    {
      SECTION("unbatched")
      {
        auto a = R5::empty(DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
      }

      SECTION("batched")
      {
        auto a = R5::empty(B, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
      }
    }

    SECTION("zeros")
    {
      SECTION("unbatched")
      {
        auto a = R5::zeros(DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::zeros_like(a)));
      }

      SECTION("batched")
      {
        auto a = R5::zeros(B, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::zeros_like(a)));
      }
    }

    SECTION("ones")
    {
      SECTION("unbatched")
      {
        auto a = R5::ones(DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::ones_like(a)));
      }

      SECTION("batched")
      {
        auto a = R5::ones(B, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::ones_like(a)));
      }
    }

    SECTION("full")
    {
      SECTION("unbatched")
      {
        Real init = 3.3;
        auto a = R5::full(init, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::full_like(a, init)));
      }

      SECTION("batched")
      {
        Real init = 2.22;
        auto a = R5::full(B, init, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == R5::const_base_dim);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == R5::const_base_sizes);
        REQUIRE(a.base_storage() == R5::const_base_storage);
        REQUIRE(at::allclose(a, at::full_like(a, init)));
      }
    }
  }
}
