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

#include <map>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "neml2/csrc/dispatchers/batch_chunk.h"

#include "test_util.h"

using namespace neml2::aoti;

int
main()
{
  const auto dbl = at::TensorOptions().dtype(at::kDouble);

  // infer_batch_size: the dim-0 extent of the fully-batched entry; broadcast /
  // unbatched entries do not raise it.
  {
    std::map<std::string, at::Tensor> m;
    m.emplace("x", at::arange(10 * 6, dbl).reshape({10, 6}));
    m.emplace("g", at::ones({1}, dbl)); // broadcast (dim-0 size 1)
    m.emplace("s", at::ones({}, dbl));  // scalar (rank 0)
    NEML2_CHECK(infer_batch_size(m) == 10);
  }
  {
    std::map<std::string, at::Tensor> m;
    m.emplace("g", at::ones({1}, dbl));
    NEML2_CHECK(infer_batch_size(m) == 1); // nothing batched
  }

  // slice_batch: narrow the batched entry; pass broadcast/unbatched through.
  {
    std::map<std::string, at::Tensor> m;
    auto x = at::arange(10 * 6, dbl).reshape({10, 6});
    auto g = at::arange(1, dbl); // (1,)
    m.emplace("x", x);
    m.emplace("g", g);

    auto sl = slice_batch(m, /*start=*/2, /*count=*/3);
    NEML2_CHECK(sl.at("x").size(0) == 3);
    NEML2_CHECK(at::equal(sl.at("x"), x.narrow(0, 2, 3)));
    // Broadcast entry passes through whole.
    NEML2_CHECK(sl.at("g").size(0) == 1);
    NEML2_CHECK(at::equal(sl.at("g"), g));
  }

  // cat_batch round-trip: split a batched-only map into uneven chunks, then
  // concatenate -> identical to the original (proves slice/cat are inverse on
  // the dynamic axis, including a final short chunk).
  {
    std::map<std::string, at::Tensor> m;
    auto x = at::randn({10, 6}, dbl);
    auto y = at::randn({10}, dbl);
    m.emplace("x", x);
    m.emplace("y", y);

    std::vector<std::map<std::string, at::Tensor>> chunks;
    const int64_t k = 4, b = 10;
    for (int64_t s = 0; s < b; s += k)
      chunks.push_back(slice_batch(m, s, std::min(k, b - s)));
    NEML2_CHECK(chunks.size() == 3); // [0,4) [4,8) [8,10)

    auto cat = cat_batch(chunks);
    NEML2_CHECK(at::equal(cat.at("x"), x));
    NEML2_CHECK(at::equal(cat.at("y"), y));
  }

  // Single-chunk cat is a pass-through.
  {
    std::map<std::string, at::Tensor> m;
    auto x = at::randn({5, 6}, dbl);
    m.emplace("x", x);
    auto cat = cat_batch({m});
    NEML2_CHECK(at::equal(cat.at("x"), x));
  }

  // cat_batch_tensor for the assembled Jacobian path.
  {
    auto a = at::randn({4, 2, 3}, dbl);
    auto b = at::randn({6, 2, 3}, dbl);
    auto cat = cat_batch_tensor({a, b});
    NEML2_CHECK(cat.size(0) == 10);
    NEML2_CHECK(at::equal(cat.narrow(0, 0, 4), a));
    NEML2_CHECK(at::equal(cat.narrow(0, 4, 6), b));
  }

  // cat_batch_nested (base-shape-aware): per-(out,in) blocks, where a batched
  // block (dim > out_base_ndim + in_base_ndim) is concatenated along dim 0 while
  // a batch-independent block (dim == that trail) is identical across chunks and
  // passed through unchanged. Out var "o" has base ndim 1; inputs "i"/"c" too.
  {
    const std::map<std::string, int64_t> out_nd{{"o", 1}};
    const std::map<std::string, int64_t> in_nd{{"i", 1}, {"c", 1}};
    auto a = at::randn({4, 2, 3}, dbl); // batched (o,i) block, chunk 0
    auto b = at::randn({6, 2, 3}, dbl); // batched (o,i) block, chunk 1
    auto bi = at::randn({2, 3}, dbl);   // batch-independent (o,c) block

    std::map<std::string, std::map<std::string, at::Tensor>> c0, c1;
    c0["o"]["i"] = a;
    c0["o"]["c"] = bi;
    c1["o"]["i"] = b;
    c1["o"]["c"] = bi; // identical across chunks (batch-independent)

    auto cat = cat_batch_nested({c0, c1}, out_nd, in_nd);
    // Batched pair: concatenated to the full batch.
    NEML2_CHECK(cat.at("o").at("i").size(0) == 10);
    NEML2_CHECK(at::equal(cat.at("o").at("i").narrow(0, 0, 4), a));
    NEML2_CHECK(at::equal(cat.at("o").at("i").narrow(0, 4, 6), b));
    // Batch-independent pair: passed through at its natural (2, 3), not stacked.
    NEML2_CHECK(cat.at("o").at("c").dim() == 2);
    NEML2_CHECK(at::equal(cat.at("o").at("c"), bi));

    // Single-chunk fast path: returned verbatim.
    auto one = cat_batch_nested({c0}, out_nd, in_nd);
    NEML2_CHECK(at::equal(one.at("o").at("i"), a));
    NEML2_CHECK(at::equal(one.at("o").at("c"), bi));
  }

  // to_device_nested: a cpu->cpu move is a no-op that preserves every block.
  {
    std::map<std::string, std::map<std::string, at::Tensor>> j;
    auto blk = at::randn({3, 2, 3}, dbl);
    j["o"]["i"] = blk;
    auto moved = to_device_nested(j, at::kCPU);
    NEML2_CHECK(at::equal(moved.at("o").at("i"), blk));
  }

  return 0;
}
