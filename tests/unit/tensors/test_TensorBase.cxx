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

#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/sign.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/abs.h"

using namespace neml2;

TEST_CASE("TensorBase", "[tensors]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2};             // batch shape
  Size Bn = B.size();                       // batch dimension
  TensorShape D = {2, 5, 1, 3};             // base shape
  Size Dn = D.size();                       // base dimension
  Size n = Bn + Dn;                         // total dimension
  TensorShape BD = utils::add_shapes(B, D); // total shape
  Size L = utils::storage_size(D);          // base storage

  SECTION("class TensorBase")
  {
    SECTION("TensorBase")
    {
      SECTION("default")
      {
        auto a = Tensor();
        REQUIRE(!a.defined());
      }

      SECTION("Tensor and batch dimension")
      {
        auto a = Tensor(at::zeros(BD, DTO), B.size());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
      }

      SECTION("copy")
      {
        auto a = Tensor(at::zeros(BD, DTO), B.size());
        auto b = Tensor(a);
        REQUIRE(b.dim() == a.dim());
        REQUIRE(b.batch_dim() == a.batch_dim());
        REQUIRE(b.base_dim() == a.base_dim());
        REQUIRE(b.sizes() == a.sizes());
        REQUIRE(b.batch_sizes() == a.batch_sizes());
        REQUIRE(b.base_sizes() == a.base_sizes());
        REQUIRE(b.base_storage() == a.base_storage());
      }
    }

    SECTION("empty")
    {
      SECTION("unbatched")
      {
        auto a = Tensor::empty(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
      }

      SECTION("batched")
      {
        auto a = Tensor::empty(B, D, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
      }
    }

    SECTION("empty_like")
    {
      auto a = Tensor::empty(B, D, DTO);
      auto b = Tensor::empty_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
    }

    SECTION("zeros")
    {
      SECTION("unbatched")
      {
        auto a = Tensor::zeros(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::zeros_like(a)));
      }

      SECTION("batched")
      {
        auto a = Tensor::zeros(B, D, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::zeros_like(a)));
      }
    }

    SECTION("zeros_like")
    {
      auto a = Tensor::empty(B, D, DTO);
      auto b = Tensor::zeros_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(at::allclose(b, at::zeros_like(b)));
    }

    SECTION("ones")
    {
      SECTION("unbatched")
      {
        auto a = Tensor::ones(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::ones_like(a)));
      }

      SECTION("batched")
      {
        auto a = Tensor::ones(B, D, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::ones_like(a)));
      }
    }

    SECTION("ones_like")
    {
      auto a = Tensor::empty(B, D, DTO);
      auto b = Tensor::ones_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(at::allclose(b, at::ones_like(b)));
    }

    SECTION("full")
    {
      SECTION("unbatched")
      {
        Real init = 4.3;
        auto a = Tensor::full(D, init, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::full_like(a, init)));
      }

      SECTION("batched")
      {
        Real init = 2999;
        auto a = Tensor::full(B, D, init, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(at::allclose(a, at::full_like(a, init)));
      }
    }

    SECTION("full_like")
    {
      Real init = -3.2;
      auto a = Tensor::empty(B, D, DTO);
      auto b = Tensor::full_like(a, init);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(at::allclose(b, init * at::ones_like(b)));
    }

    SECTION("identity")
    {
      SECTION("unbatched")
      {
        Size n = 21;
        TensorShape Deye = {n, n};
        auto a = Tensor::identity(n, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == 2);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == 2);
        REQUIRE(a.sizes() == Deye);
        REQUIRE(a.batch_sizes() == TensorShape{});
        REQUIRE(a.base_sizes() == Deye);
        REQUIRE(a.base_storage() == n * n);
        REQUIRE(at::allclose(a, at::eye(n, DTO)));
      }

      SECTION("batched")
      {
        Size n = 33;
        TensorShape Deye = {n, n};
        TensorShape BDeye = utils::add_shapes(B, Deye);
        auto a = Tensor::identity(B, n, DTO);
        REQUIRE(a.batched());
        REQUIRE(a.dim() == Bn + 2);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == 2);
        REQUIRE(a.sizes() == BDeye);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == Deye);
        REQUIRE(a.base_storage() == n * n);
        REQUIRE(at::allclose(a, at::eye(n, DTO).expand(utils::add_shapes(B, -1, -1))));
      }
    }

    SECTION("linspace")
    {
      Size nstep = 101;
      Size dim = 2;
      TensorShape B_new = B;
      B_new.insert(B_new.begin() + dim, nstep);
      TensorShape BD_new = utils::add_shapes(B_new, D);

      auto a = Tensor::full(B, D, -5.5, DTO);
      auto b = Tensor::full(B, D, 123, DTO);
      auto c = Tensor::linspace(a, b, nstep, dim);
      REQUIRE(c.dim() == n + 1);
      REQUIRE(c.batch_dim() == Bn + 1);
      REQUIRE(c.base_dim() == Dn);
      REQUIRE(c.sizes() == BD_new);
      REQUIRE(c.batch_sizes() == B_new);
      REQUIRE(c.base_sizes() == D);
      REQUIRE(c.base_storage() == L);
    }

    SECTION("batch_index")
    {
      auto a = at::rand({5, 3, 2, 1, 3}, DTO);
      auto b = Tensor(a, 3);
      indexing::TensorIndices i1 = {0};
      indexing::TensorIndices i2 = {2, 1};
      indexing::TensorIndices i3 = {indexing::Slice(), indexing::Slice(0, 2), 1};
      indexing::TensorIndices i4a = {2, indexing::Ellipsis, indexing::Slice(), indexing::Slice()};
      indexing::TensorIndices i4b = {2, indexing::Ellipsis};
      REQUIRE(at::allclose(a.index(i1), b.batch_index(i1)));
      REQUIRE(at::allclose(a.index(i2), b.batch_index(i2)));
      REQUIRE(at::allclose(a.index(i3), b.batch_index(i3)));
      REQUIRE(at::allclose(a.index(i4a), b.batch_index(i4b)));
    }

    SECTION("base_index")
    {
      auto a = at::rand({5, 3, 2, 1, 3}, DTO);
      auto b = Tensor(a, 3);
      indexing::TensorIndices i1a = {indexing::Slice(), indexing::Slice(), indexing::Slice(), 0};
      indexing::TensorIndices i1b = {0};
      indexing::TensorIndices i2a = {
          indexing::Slice(), indexing::Slice(), indexing::Slice(), indexing::Ellipsis};
      indexing::TensorIndices i2b = {indexing::Ellipsis};
      indexing::TensorIndices i3a = {indexing::Ellipsis};
      indexing::TensorIndices i3b = {indexing::Ellipsis};
      indexing::TensorIndices i4a = {indexing::Slice(),
                                     indexing::Slice(),
                                     indexing::Slice(),
                                     indexing::Slice(),
                                     indexing::Slice(1, indexing::None)};
      indexing::TensorIndices i4b = {indexing::Slice(), indexing::Slice(1, indexing::None)};
      REQUIRE(at::allclose(a.index(i1a), b.base_index(i1b)));
      REQUIRE(at::allclose(a.index(i2a), b.base_index(i2b)));
      REQUIRE(at::allclose(a.index(i3a), b.base_index(i3b)));
      REQUIRE(at::allclose(a.index(i4a), b.base_index(i4b)));
    }

    SECTION("batch_index_put")
    {
      auto a = at::rand({5, 3, 2, 1, 3}, DTO);
      auto b = Tensor(a, 3);
      auto c = at::rand({3, 2, 1, 3}, DTO);
      indexing::TensorIndices ia = {2, indexing::Ellipsis, indexing::Slice(), indexing::Slice()};
      indexing::TensorIndices ib = {2, indexing::Ellipsis};
      b.batch_index_put_(ib, c);
      REQUIRE(at::allclose(a.index(ia), c));
    }

    SECTION("base_index_put")
    {
      auto a = at::rand({5, 3, 2, 1, 3}, DTO);
      auto b = Tensor(a, 3);
      auto c = at::rand({5, 3, 2, 1, 2}, DTO);
      indexing::TensorIndices ia = {indexing::Slice(),
                                    indexing::Slice(),
                                    indexing::Slice(),
                                    indexing::Slice(),
                                    indexing::Slice(1, indexing::None)};
      indexing::TensorIndices ib = {indexing::Slice(), indexing::Slice(1, indexing::None)};
      b.base_index_put_(ib, c);
      REQUIRE(at::allclose(a.index(ia), c));
    }

    SECTION("batch_expand")
    {
      TensorShape s0 = {5, 1, 2, 1, 5};
      TensorShape s = {5, 8, 2, 2, 5};
      auto a = Tensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = a.batch_expand(s);
      REQUIRE(b.batch_sizes() == s);
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(at::sum(a - b).item<Real>() == Catch::Approx(0));
    }

    SECTION("base_expand")
    {
      TensorShape s0 = {2, 1, 3, 1, 3};
      TensorShape s = {2, 7, 3, 1, 3};
      auto a = Tensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = a.base_expand(s);
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == s);
      // This is fun, as a and b are NOT broadcastable based on our broadcasting rules for batched
      // tensors because they have different base shapes. However, they _should_ be broadcastable
      // based on libTorch's original broadcasting rules. So we need to interpret them as
      // ATensors first before we can compute a - b. This is the correct behavior.
      REQUIRE(at::sum(ATensor(a) - ATensor(b)).item<Real>() == Catch::Approx(0));
    }

    SECTION("batch_expand_as")
    {
      TensorShape s0 = {5, 1, 2, 1, 5};
      TensorShape s = {5, 8, 2, 2, 5};
      auto a = Tensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = Tensor::full(s, {5}, 3.33, DTO); // base shapes can differ!
      auto c = a.batch_expand_as(b);
      REQUIRE(c.batch_sizes() == s);
      REQUIRE(c.base_sizes() == a.base_sizes());
    }

    SECTION("base_expand_as")
    {
      TensorShape s0 = {2, 1, 3, 1, 3};
      TensorShape s = {2, 7, 3, 1, 3};
      auto a = Tensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = Tensor::full({3, 2, 1}, s, 3.33, DTO); // batch shapes can differ!
      auto c = a.base_expand_as(b);
      REQUIRE(c.batch_sizes() == a.batch_sizes());
      REQUIRE(c.base_sizes() == s);
    }

    SECTION("batch_expand_copy")
    {
      TensorShape s0 = {5, 1, 2, 1, 5};
      TensorShape s = {5, 8, 2, 2, 5};
      auto a = Tensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = a.batch_expand_copy(s);
      REQUIRE(b.batch_sizes() == s);
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(at::sum(a - b).item<Real>() == Catch::Approx(0));
    }

    SECTION("base_expand_copy")
    {
      TensorShape s0 = {2, 1, 3, 1, 3};
      TensorShape s = {2, 7, 3, 1, 3};
      auto a = Tensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = a.base_expand_copy(s);
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == s);
      // This is fun, as a and b are NOT broadcastable based on our broadcasting rules for batched
      // tensors because they have different base shapes. However, they _should_ be broadcastable
      // based on libTorch's original broadcasting rules. So we need to interpret them as
      // ATensors first before we can compute a - b. This is the correct behavior.
      REQUIRE(at::sum(ATensor(a) - ATensor(b)).item<Real>() == Catch::Approx(0));
    }

    SECTION("batch_unsqueeze")
    {
      auto a = Tensor::full({2, 5}, {3, 3}, 5.25324, DTO);

      auto a1 = a.batch_unsqueeze(0);
      REQUIRE(a1.batch_sizes() == TensorShape{1, 2, 5});
      REQUIRE(a1.base_sizes() == a.base_sizes());

      auto a2 = a.batch_unsqueeze(1);
      REQUIRE(a2.batch_sizes() == TensorShape{2, 1, 5});
      REQUIRE(a2.base_sizes() == a.base_sizes());

      auto a3 = a.batch_unsqueeze(-2);
      REQUIRE(a3.batch_sizes() == TensorShape{2, 1, 5});
      REQUIRE(a3.base_sizes() == a.base_sizes());
    }

    SECTION("base_unsqueeze")
    {
      auto a = Tensor::full({2, 5}, {3, 3}, 5.25324, DTO);

      auto a1 = a.base_unsqueeze(0);
      REQUIRE(a1.batch_sizes() == a.batch_sizes());
      REQUIRE(a1.base_sizes() == TensorShape{1, 3, 3});

      auto a2 = a.base_unsqueeze(1);
      REQUIRE(a2.batch_sizes() == a.batch_sizes());
      REQUIRE(a2.base_sizes() == TensorShape{3, 1, 3});

      auto a3 = a.base_unsqueeze(-2);
      REQUIRE(a3.batch_sizes() == a.batch_sizes());
      REQUIRE(a3.base_sizes() == TensorShape{3, 1, 3});
    }

    SECTION("batch_transpose")
    {
      auto a = Tensor::full({2, 3, 5, 2}, {3, 3}, 5.25324, DTO);
      auto b = a.batch_transpose(1, 3);
      REQUIRE(b.batch_sizes() == TensorShape{2, 2, 5, 3});
      REQUIRE(b.base_sizes() == a.base_sizes());
    }

    SECTION("base_transpose")
    {
      auto a = Tensor::full({3, 3}, {5, 3, 5, 2}, 5.25324, DTO);
      auto b = a.base_transpose(0, 3);
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == TensorShape{2, 3, 5, 5});
    }
  }

  SECTION("operator+")
  {
    auto a = Tensor::create({{3.1, 2.2}, {2.2, -1.1}}, DTO);
    auto b = Tensor::full({}, {2, 2}, 2.0, DTO);
    auto c = Tensor::create({{5.1, 4.2}, {4.2, 0.9}}, DTO);
    REQUIRE(at::allclose(a + 2.0, c));
    REQUIRE(at::allclose(a.batch_expand(B) + 2.0, c.batch_expand(B)));
    REQUIRE(at::allclose(2.0 + a, c));
    REQUIRE(at::allclose(2.0 + a.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a + b, c));
    REQUIRE(at::allclose(a + b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) + b, c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) + b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator-")
  {
    auto a = Tensor::create({{3.1, 2.2}, {2.2, -1.1}}, DTO);
    auto b = Tensor::full({}, {2, 2}, 2.0, DTO);
    auto c = Tensor::create({{1.1, 0.2}, {0.2, -3.1}}, DTO);
    REQUIRE(at::allclose(a - 2.0, c));
    REQUIRE(at::allclose(a.batch_expand(B) - 2.0, c.batch_expand(B)));
    REQUIRE(at::allclose(2.0 - a, -c));
    REQUIRE(at::allclose(2.0 - a.batch_expand(B), -c.batch_expand(B)));
    REQUIRE(at::allclose(a - b, c));
    REQUIRE(at::allclose(a - b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) - b, c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) - b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator*")
  {
    auto a = Tensor::create({{3.1, 2.2}, {2.2, -1.1}}, DTO);
    auto b = Tensor::full({}, {2, 2}, 2.0, DTO);
    auto c = Tensor::create({{6.2, 4.4}, {4.4, -2.2}}, DTO);
    REQUIRE(at::allclose(a * 2.0, c));
    REQUIRE(at::allclose(a.batch_expand(B) * 2.0, c.batch_expand(B)));
    REQUIRE(at::allclose(2.0 * a, c));
    REQUIRE(at::allclose(2.0 * a.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator/")
  {
    auto a = Tensor::create({{3.1, 2.2}, {2.2, -1.1}}, DTO);
    auto b = Tensor::full({}, {2, 2}, 2.0, DTO);
    auto c = Tensor::create({{1.55, 1.1}, {1.1, -0.55}}, DTO);
    auto cinv = 1.0 / Tensor::create({{1.55, 1.1}, {1.1, -0.55}}, DTO);
    REQUIRE(at::allclose(a / 2.0, c));
    REQUIRE(at::allclose(a.batch_expand(B) / 2.0, c.batch_expand(B)));
    REQUIRE(at::allclose(2.0 / a, cinv));
    REQUIRE(at::allclose(2.0 / a.batch_expand(B), cinv.batch_expand(B)));
    REQUIRE(at::allclose(a / b, c));
    REQUIRE(at::allclose(a / b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) / b, c.batch_expand(B)));
    REQUIRE(at::allclose(a.batch_expand(B) / b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("pow")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::full({}, {2, 2}, 2.0, DTO);
    auto c = Tensor::create({{9.0, 4.0}, {4.0, 1.21}}, DTO);
    REQUIRE(at::allclose(pow(a, 2.0), c));
    REQUIRE(at::allclose(pow(a.batch_expand(B), 2.0), c.batch_expand(B)));
  }

  SECTION("sign")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::create({{1.0, 1.0}, {1.0, -1.0}}, DTO);
    REQUIRE(at::allclose(sign(a), b));
    REQUIRE(at::allclose(sign(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("heaviside")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::create({{1.0, 1.0}, {1.0, 0.0}}, DTO);
    REQUIRE(at::allclose(heaviside(a), b));
    REQUIRE(at::allclose(heaviside(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("macaulay")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::create({{3.0, 2.0}, {2.0, 0.0}}, DTO);
    REQUIRE(at::allclose(macaulay(a), b));
    REQUIRE(at::allclose(macaulay(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("sqrt")
  {
    auto a = Tensor::create({{4.0, 9.0}, {25.0, 64.0}}, DTO);
    auto b = Tensor::create({{2.0, 3.0}, {5.0, 8.0}}, DTO);
    REQUIRE(at::allclose(sqrt(a), b));
    REQUIRE(at::allclose(sqrt(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("exp")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::create(
        {{20.085536923187668, 7.38905609893065}, {7.38905609893065, 0.33287108369807955}}, DTO);
    REQUIRE(at::allclose(exp(a), b));
    REQUIRE(at::allclose(exp(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("abs")
  {
    auto a = Tensor::create({{3.0, 2.0}, {2.0, -1.1}}, DTO);
    auto b = Tensor::create({{3.0, 2.0}, {2.0, 1.1}}, DTO);
    REQUIRE(at::allclose(abs(a), b));
    REQUIRE(at::allclose(abs(a.batch_expand(B)), b.batch_expand(B)));
  }
}
