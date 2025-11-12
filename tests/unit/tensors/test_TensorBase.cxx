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

#include <ATen/Context.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/tensors/indexing.h"
#include "neml2/tensors/shape_utils.h"
#include "unit/tensors/generators.h"
#include "neml2/tensors/Tensor.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("TensorBase", "[tensors]")
{
  at::manual_seed(42);

  SECTION("constructors")
  {
    SECTION("default")
    {
      auto a = Tensor();
      REQUIRE(!a.defined());
    }

    SECTION("ATensor, dynamic dimension, intermediate dimension")
    {
      auto cfg = test::generate_tensor_config();
      auto shape = test::generate_tensor_shape();
      DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
      {
        auto a = test::generate_random_tensor(cfg, shape);
        auto b = Tensor(a, shape.dynamic_dim, shape.intmd_dim);
        REQUIRE(test::match_tensor_config(b, cfg));
        REQUIRE(test::match_tensor_shape(b, shape));
        REQUIRE_THAT(a, test::allclose(ATensor(b)));
      }
    }

    SECTION("ATensor, dynamic shape, intermediate dimension")
    {
      auto cfg = test::generate_tensor_config();
      auto shape = test::generate_tensor_shape();
      DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
      {
        auto a = test::generate_random_tensor(cfg, shape);
        auto b = Tensor(a, shape.dynamic_sizes, shape.intmd_dim);
        REQUIRE(test::match_tensor_config(b, cfg));
        REQUIRE(test::match_tensor_shape(b, shape));
        REQUIRE_THAT(a, test::allclose(ATensor(b)));
      }
    }
  }

  SECTION("empty_like")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = Tensor::empty_like(a);
      REQUIRE(test::match_tensor_config(b, cfg));
      REQUIRE(test::match_tensor_shape(b, shape));
    }
  }

  SECTION("zeros_like")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = Tensor::zeros_like(a);
      auto c = at::zeros_like(a);
      REQUIRE(test::match_tensor_config(b, cfg));
      REQUIRE(test::match_tensor_shape(b, shape));
      REQUIRE_THAT(b, test::allclose(c));
    }
  }

  SECTION("ones_like")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = Tensor::ones_like(a);
      auto c = at::ones_like(a);
      REQUIRE(test::match_tensor_config(b, cfg));
      REQUIRE(test::match_tensor_shape(b, shape));
      REQUIRE_THAT(b, test::allclose(c));
    }
  }

  SECTION("full_like")
  {
    auto cfg = test::generate_tensor_config();
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = Tensor::full_like(a, 3.5);
      auto c = at::full_like(a, 3.5);
      REQUIRE(test::match_tensor_config(b, cfg));
      REQUIRE(test::match_tensor_shape(b, shape));
      REQUIRE_THAT(b, test::allclose(c));
    }
  }

  SECTION("rand_like")
  {
    auto cfg = test::generate_tensor_config(test::fp_dtypes());
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(cfg.desc() << " " << shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = Tensor::rand_like(a);
      auto c = at::rand_like(a);
      REQUIRE(test::match_tensor_config(b, cfg));
      REQUIRE(test::match_tensor_shape(b, shape));
    }
  }

  SECTION("dim")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      REQUIRE(a.dynamic_dim() == shape.dynamic_dim);
      REQUIRE(a.intmd_dim() == shape.intmd_dim);
      REQUIRE(a.base_dim() == shape.base_dim);
      REQUIRE(a.batch_dim() == shape.batch_dim);
      REQUIRE(a.static_dim() == shape.static_dim);
    }
  }

  SECTION("sizes")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      REQUIRE(a.dynamic_sizes() == TensorShapeRef(shape.dynamic_sizes));
      REQUIRE(a.intmd_sizes() == TensorShapeRef(shape.intmd_sizes));
      REQUIRE(a.base_sizes() == TensorShapeRef(shape.base_sizes));
      REQUIRE(a.batch_sizes() == TensorShapeRef(shape.batch_sizes));
      REQUIRE(a.static_sizes() == TensorShapeRef(shape.static_sizes));
    }
  }

  SECTION("size")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::generate_tensor_shape();
    DYNAMIC_SECTION(shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);

      for (Size i = -shape.dynamic_dim; i < shape.dynamic_dim; i++)
        REQUIRE(a.dynamic_size(i) ==
                shape.dynamic_sizes[utils::normalize_dim(i, 0, shape.dynamic_dim)]);

      for (Size i = -shape.intmd_dim; i < shape.intmd_dim; i++)
        REQUIRE(a.intmd_size(i) == shape.intmd_sizes[utils::normalize_dim(i, 0, shape.intmd_dim)]);

      for (Size i = -shape.base_dim; i < shape.base_dim; i++)
        REQUIRE(a.base_size(i) == shape.base_sizes[utils::normalize_dim(i, 0, shape.base_dim)]);

      for (Size i = -shape.batch_dim; i < shape.batch_dim; i++)
        REQUIRE(a.batch_size(i) == shape.batch_sizes[utils::normalize_dim(i, 0, shape.batch_dim)]);

      for (Size i = -shape.static_dim; i < shape.static_dim; i++)
        REQUIRE(a.static_size(i) ==
                shape.static_sizes[utils::normalize_dim(i, 0, shape.static_dim)]);
    }
  }

  SECTION("index")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 3}, {2, 1, 4}, {1, 2, 3});

    SECTION("ellipsis")
    {
      indexing::TensorIndices i = {indexing::Ellipsis};
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);

      auto b = a.dynamic_index(i);
      auto c = a.intmd_index(i);
      auto d = a.base_index(i);

      REQUIRE_THAT(a, test::allclose(b));
      REQUIRE_THAT(a, test::allclose(c));
      REQUIRE_THAT(a, test::allclose(d));
    }

    SECTION("slice")
    {
      indexing::TensorIndices i = {indexing::Slice(), indexing::Slice(), indexing::Slice(1, 3)};
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);

      auto b = a.dynamic_index(i);
      auto c = a.intmd_index(i);
      auto d = a.base_index(i);

      auto ib = i;
      ib.insert(ib.end(), shape.static_dim, indexing::Slice());
      auto ic = i;
      ic.insert(ic.begin(), shape.dynamic_dim, indexing::Slice());
      ic.insert(ic.end(), shape.base_dim, indexing::Slice());
      auto id = i;
      id.insert(id.begin(), shape.batch_dim, indexing::Slice());

      REQUIRE_THAT(a.index(ib), test::allclose(ATensor(b)));
      REQUIRE_THAT(a.index(ic), test::allclose(ATensor(c)));
      REQUIRE_THAT(a.index(id), test::allclose(ATensor(d)));
    }

    SECTION("integer")
    {
      indexing::TensorIndices i = {indexing::Slice(), indexing::Slice(), 1};
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);

      auto b = a.dynamic_index(i);
      auto c = a.intmd_index(i);
      auto d = a.base_index(i);

      auto ib = i;
      ib.insert(ib.end(), shape.static_dim, indexing::Slice());
      auto ic = i;
      ic.insert(ic.begin(), shape.dynamic_dim, indexing::Slice());
      ic.insert(ic.end(), shape.base_dim, indexing::Slice());
      auto id = i;
      id.insert(id.begin(), shape.batch_dim, indexing::Slice());

      REQUIRE_THAT(a.index(ib), test::allclose(ATensor(b)));
      REQUIRE_THAT(a.index(ic), test::allclose(ATensor(c)));
      REQUIRE_THAT(a.index(id), test::allclose(ATensor(d)));
    }

    SECTION("mixed")
    {
      indexing::TensorIndices i = {indexing::Slice(0, 1), indexing::Ellipsis, 1};
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);

      auto b = a.dynamic_index(i);
      auto c = a.intmd_index(i);
      auto d = a.base_index(i);

      auto ib = i;
      ib.insert(ib.end(), shape.static_dim, indexing::Slice());
      auto ic = i;
      ic.insert(ic.begin(), shape.dynamic_dim, indexing::Slice());
      ic.insert(ic.end(), shape.base_dim, indexing::Slice());
      auto id = i;
      id.insert(id.begin(), shape.batch_dim, indexing::Slice());

      REQUIRE_THAT(a.index(ib), test::allclose(ATensor(b)));
      REQUIRE_THAT(a.index(ic), test::allclose(ATensor(c)));
      REQUIRE_THAT(a.index(id), test::allclose(ATensor(d)));
    }
  }

  SECTION("slice")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 3}, {2, 1, 4}, {1, 2, 3});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_slice(1, indexing::Slice(0, 2));
    auto c = a.intmd_slice(2, indexing::Slice(1, 4));
    auto d = a.base_slice(0, indexing::Slice(0, 1));
    REQUIRE_THAT(a.slice(1, 0, 2), test::allclose(ATensor(b)));
    REQUIRE_THAT(a.slice(2 + shape.dynamic_dim, 1, 4), test::allclose(ATensor(c)));
    REQUIRE_THAT(a.slice(0 + shape.batch_dim, 0, 1), test::allclose(ATensor(d)));
  }

  SECTION("index_put_")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 3}, {2, 1, 4}, {1, 2, 3});
    indexing::TensorIndices i = {indexing::Ellipsis, 0, indexing::Slice(1, 3)};
    auto a = test::generate_random_tensor<Tensor>(cfg, shape);

    SECTION("ATensor")
    {
      auto ib = i;
      ib.insert(ib.end(), shape.static_dim, indexing::Slice());
      auto b = a.clone();
      auto vb = at::rand_like(a.index(ib));
      b.dynamic_index_put_(i, vb);

      auto ic = i;
      ic.insert(ic.begin(), shape.dynamic_dim, indexing::Slice());
      ic.insert(ic.end(), shape.base_dim, indexing::Slice());
      auto c = a.clone();
      auto vc = at::rand_like(a.index(ic));
      c.intmd_index_put_(i, vc);

      auto id = i;
      id.insert(id.begin(), shape.batch_dim, indexing::Slice());
      auto d = a.clone();
      auto vd = at::rand_like(a.index(id));
      d.base_index_put_(i, vd);

      REQUIRE_THAT(b.index(ib), test::allclose(ATensor(vb)));
      REQUIRE_THAT(c.index(ic), test::allclose(ATensor(vc)));
      REQUIRE_THAT(d.index(id), test::allclose(ATensor(vd)));
    }

    SECTION("CScalar")
    {
      auto b = a.clone();
      b.dynamic_index_put_(i, 5.5);

      auto c = a.clone();
      c.intmd_index_put_(i, 5.5);

      auto d = a.clone();
      d.base_index_put_(i, 5.5);

      auto ib = i;
      ib.insert(ib.end(), shape.static_dim, indexing::Slice());
      auto ic = i;
      ic.insert(ic.begin(), shape.dynamic_dim, indexing::Slice());
      ic.insert(ic.end(), shape.base_dim, indexing::Slice());
      auto id = i;
      id.insert(id.begin(), shape.batch_dim, indexing::Slice());

      REQUIRE_THAT(b.index(ib), test::allclose(at::full_like(a.index(ib), 5.5)));
      REQUIRE_THAT(c.index(ic), test::allclose(at::full_like(a.index(ic), 5.5)));
      REQUIRE_THAT(d.index(id), test::allclose(at::full_like(a.index(id), 5.5)));
    }
  }

  SECTION("expand")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 1}, {1, 4}, {2, 1});
    auto a = test::generate_random_tensor<Tensor>(cfg, shape);

    auto b = a.dynamic_expand({3, 5});
    auto c = a.intmd_expand({5, 1, 4});
    auto d = a.base_expand({1, 2, 2});
    auto e = a.batch_expand({3, 5}, {5, 1, 4});
    auto f = a.static_expand({2, 1, 4}, {2, 2, 1});

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{3, 5});
    REQUIRE(b.intmd_sizes() == TensorShapeRef{1, 4});
    REQUIRE(b.base_sizes() == TensorShapeRef{2, 1});

    REQUIRE(c.dynamic_sizes() == TensorShapeRef{3, 1});
    REQUIRE(c.intmd_sizes() == TensorShapeRef{5, 1, 4});
    REQUIRE(c.base_sizes() == TensorShapeRef{2, 1});

    REQUIRE(d.dynamic_sizes() == TensorShapeRef{3, 1});
    REQUIRE(d.intmd_sizes() == TensorShapeRef{1, 4});
    REQUIRE(d.base_sizes() == TensorShapeRef{1, 2, 2});

    REQUIRE(e.dynamic_sizes() == TensorShapeRef{3, 5});
    REQUIRE(e.intmd_sizes() == TensorShapeRef{5, 1, 4});
    REQUIRE(e.base_sizes() == TensorShapeRef{2, 1});

    REQUIRE(f.dynamic_sizes() == TensorShapeRef{3, 1});
    REQUIRE(f.intmd_sizes() == TensorShapeRef{2, 1, 4});
    REQUIRE(f.base_sizes() == TensorShapeRef{2, 2, 1});

    b = a.dynamic_expand(5, 1);
    c = a.intmd_expand(5, 0);
    d = a.base_expand(3, 1);

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{3, 5});
    REQUIRE(b.intmd_sizes() == TensorShapeRef{1, 4});
    REQUIRE(b.base_sizes() == TensorShapeRef{2, 1});

    REQUIRE(c.dynamic_sizes() == TensorShapeRef{3, 1});
    REQUIRE(c.intmd_sizes() == TensorShapeRef{5, 4});
    REQUIRE(c.base_sizes() == TensorShapeRef{2, 1});

    REQUIRE(d.dynamic_sizes() == TensorShapeRef{3, 1});
    REQUIRE(d.intmd_sizes() == TensorShapeRef{1, 4});
    REQUIRE(d.base_sizes() == TensorShapeRef{2, 3});
  }

  SECTION("expand_as")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 1}, {1, 2}, {1});
    auto target = Tensor::empty({2, 3, 2}, {1, 3, 2}, {3, 3, 2}, cfg.options);

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_expand_as(target);
    auto c = a.intmd_expand_as(target);
    auto d = a.base_expand_as(target);
    auto e = a.batch_expand_as(target);
    auto f = a.static_expand_as(target);

    REQUIRE(b.dynamic_sizes() == target.dynamic_sizes());
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == target.intmd_sizes());
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == target.base_sizes());

    REQUIRE(e.dynamic_sizes() == target.dynamic_sizes());
    REQUIRE(e.intmd_sizes() == target.intmd_sizes());
    REQUIRE(e.base_sizes() == a.base_sizes());

    REQUIRE(f.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(f.intmd_sizes() == target.intmd_sizes());
    REQUIRE(f.base_sizes() == target.base_sizes());
  }

  SECTION("reshape")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 4}, {4, 2}, {3, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_reshape({6, 2, 1});
    auto c = a.intmd_reshape({1, 8, 1});
    auto d = a.base_reshape({6, 1});
    auto e = a.batch_reshape({2, 3, 2}, {8});
    auto f = a.static_reshape({8}, {2, 3});

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{6, 2, 1});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());
    REQUIRE_THAT(a.flatten(), test::allclose(b.flatten()));

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{1, 8, 1});
    REQUIRE(c.base_sizes() == a.base_sizes());
    REQUIRE_THAT(a.flatten(), test::allclose(c.flatten()));

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{6, 1});
    REQUIRE_THAT(a.flatten(), test::allclose(d.flatten()));

    REQUIRE(e.dynamic_sizes() == TensorShapeRef{2, 3, 2});
    REQUIRE(e.intmd_sizes() == TensorShapeRef{8});
    REQUIRE(e.base_sizes() == a.base_sizes());
    REQUIRE_THAT(a.flatten(), test::allclose(e.flatten()));

    REQUIRE(f.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(f.intmd_sizes() == TensorShapeRef{8});
    REQUIRE(f.base_sizes() == TensorShapeRef{2, 3});
    REQUIRE_THAT(a.flatten(), test::allclose(f.flatten()));
  }

  SECTION("squeeze")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 1, 4}, {2, 3, 1}, {1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_squeeze(1);
    auto c = a.intmd_squeeze(2);
    auto d = a.base_squeeze(0);

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{3, 4});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{2, 3});
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{2});
  }

  SECTION("unsqueeze")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 4}, {2, 3}, {1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_unsqueeze(0);
    auto c = a.intmd_unsqueeze(1);
    auto d = a.base_unsqueeze(2);

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{1, 3, 4});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{2, 1, 3});
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{1, 2, 1});
  }

  SECTION("unsqueeze_n")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 4}, {2, 3}, {1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);

    auto b1 = a.dynamic_unsqueeze_n(2, 1);
    auto b2 = a.dynamic_unsqueeze_n(2, -1);

    REQUIRE(b1.dynamic_sizes() == TensorShapeRef{3, 1, 1, 4});
    REQUIRE(b1.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b1.base_sizes() == a.base_sizes());

    REQUIRE(b2.dynamic_sizes() == TensorShapeRef{3, 4, 1, 1});
    REQUIRE(b2.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b2.base_sizes() == a.base_sizes());

    auto c1 = a.intmd_unsqueeze_n(2, 1);
    auto c2 = a.intmd_unsqueeze_n(2, -1);

    REQUIRE(c1.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c1.intmd_sizes() == TensorShapeRef{2, 1, 1, 3});
    REQUIRE(c1.base_sizes() == a.base_sizes());

    REQUIRE(c2.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c2.intmd_sizes() == TensorShapeRef{2, 3, 1, 1});
    REQUIRE(c2.base_sizes() == a.base_sizes());

    auto d1 = a.base_unsqueeze_n(2, 1);
    auto d2 = a.base_unsqueeze_n(2, -1);

    REQUIRE(d1.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d1.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d1.base_sizes() == TensorShapeRef{1, 1, 1, 2});

    REQUIRE(d2.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d2.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d2.base_sizes() == TensorShapeRef{1, 2, 1, 1});
  }

  SECTION("transpose")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 4}, {2, 3, 4}, {4, 1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_transpose(0, 1);
    auto c = a.intmd_transpose(2, 0);
    auto d = a.base_transpose(-1, -2);

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{2, 3, 4});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{4, 3, 2});
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{4, 2, 1});
  }

  SECTION("movedim")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 4}, {2, 3, 4}, {4, 1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_movedim(0, 2);
    auto c = a.intmd_movedim(2, 0);
    auto d = a.base_movedim(-1, -2);

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{2, 4, 3});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{4, 2, 3});
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{4, 2, 1});
  }

  SECTION("flatten")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::GeneratedTensorShape({3, 2, 4}, {2, 3, 4}, {4, 1, 2});

    auto a = test::generate_random_tensor<Tensor>(cfg, shape);
    auto b = a.dynamic_flatten();
    auto c = a.intmd_flatten();
    auto d = a.base_flatten();
    auto e = a.batch_flatten();
    auto f = a.static_flatten();

    REQUIRE(b.dynamic_sizes() == TensorShapeRef{24});
    REQUIRE(b.intmd_sizes() == a.intmd_sizes());
    REQUIRE(b.base_sizes() == a.base_sizes());

    REQUIRE(c.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(c.intmd_sizes() == TensorShapeRef{24});
    REQUIRE(c.base_sizes() == a.base_sizes());

    REQUIRE(d.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(d.intmd_sizes() == a.intmd_sizes());
    REQUIRE(d.base_sizes() == TensorShapeRef{8});

    REQUIRE(e.dynamic_sizes() == TensorShapeRef{576});
    REQUIRE(e.intmd_sizes() == TensorShapeRef{});
    REQUIRE(e.base_sizes() == a.base_sizes());

    REQUIRE(f.dynamic_sizes() == a.dynamic_sizes());
    REQUIRE(f.intmd_sizes() == TensorShapeRef{});
    REQUIRE(f.base_sizes() == TensorShapeRef{192});
  }
}
