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

// cpp-aoti / cpp-dispatch derivatives through and w.r.t. sub-batched variables.
// argv[1]: implicit_sub_batch_ift_reduce (-d vf:s x:s x:x~1 vf:x~1 y:z), every
//   block against its closed form, all three forms present.
// argv[2]: forward_sub_batch (-d stress:strain), a batch-independent "paired"
//   block that chunked dispatch must not concatenate along its site axis.

#include <cmath>
#include <map>
#include <memory>
#include <string>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/dispatchers/DispatchedModel.h"
#include "neml2/csrc/dispatchers/SimpleScheduler.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
// NEML2_CHECK returns 1 from the enclosing function; main() forwards it.
bool
close(const at::Tensor & a, const at::Tensor & b)
{
  return a.sizes() == b.sizes() && at::allclose(a, b, /*rtol=*/1e-12, /*atol=*/1e-14);
}

int
check_reduce(const std::string & artifact_root)
{
  Model m(artifact_root, at::kCPU, at::kDouble);
  const std::map<std::string, std::map<std::string, std::string>> forms{
      {"vf", {{"s", "plain"}, {"x~1", "full"}}},
      {"x", {{"s", "plain"}, {"x~1", "full"}}},
      {"y", {{"z", "paired"}}}};
  NEML2_CHECK(m.derivative_forms() == forms);

  const int64_t b = 5, n = 4;
  const double dt = 0.1;
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  const std::map<std::string, at::Tensor> ins{{"x", at::rand({b, n}, opts)},
                                              {"x~1", at::rand({b, n}, opts)},
                                              {"s", at::rand({b}, opts)},
                                              {"t", at::full({b}, dt, opts)},
                                              {"t~1", at::zeros({b}, opts)},
                                              {"z", at::rand({b, n}, opts)}};
  const auto J = m.jacobian(ins).second;

  const auto R = at::logspace(-1.0, 0.0, n, 10.0, opts);
  const auto k = -at::logspace(0.0, 1.0, n, 10.0, opts);
  const auto w = 4.0 / 3.0 * M_PI * R.pow(3);
  const auto dx_ds = dt / (1.0 - k * dt);   // per site
  const auto dx_dxo = 1.0 / (1.0 - k * dt); // per site, diagonal

  NEML2_CHECK(close(J.at("vf").at("s"), (w * dx_ds).sum().expand({b})));
  NEML2_CHECK(close(J.at("x").at("s"), dx_ds.expand({b, n})));
  NEML2_CHECK(close(J.at("x").at("x~1"), at::diag(dx_dxo).expand({b, n, n})));
  NEML2_CHECK(close(J.at("vf").at("x~1"), (w * dx_dxo).expand({b, n})));
  NEML2_CHECK(close(J.at("y").at("z"), R.expand({b, n})));

  // Chunked dispatch matches the single-shot blocks.
  for (std::size_t c : {std::size_t{2}, std::size_t{0}})
  {
    DispatchedModel disp(artifact_root,
                         std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", c}));
    NEML2_CHECK(disp.derivative_forms() == forms);
    const auto Jd = disp.jacobian(ins).second;
    for (const auto & [o, row] : J)
      for (const auto & [i, blk] : row)
        NEML2_CHECK(close(Jd.at(o).at(i), blk));
  }
  return 0;
}

int
check_paired_batch_independent(const std::string & artifact_root)
{
  Model m(artifact_root, at::kCPU, at::kDouble);
  NEML2_CHECK(m.derivative_forms().at("stress").at("strain") == "paired");
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  const std::map<std::string, at::Tensor> ins{{"strain", at::rand({4, 5, 6}, opts)}};
  const auto ref = m.jacobian(ins).second.at("stress").at("strain");
  NEML2_CHECK(ref.sizes() == at::IntArrayRef({5, 6, 6})); // unbatched, per site
  DispatchedModel disp(artifact_root,
                       std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 2}));
  NEML2_CHECK(close(disp.jacobian(ins).second.at("stress").at("strain"), ref));
  return 0;
}
} // namespace

int
main(int argc, char ** argv)
{
  NEML2_CHECK(argc >= 3); // argv[1], argv[2] = the fixture (collection) dirs
  if (const int rc = check_reduce(std::string(argv[1]) + "/model"))
    return rc;
  return check_paired_batch_independent(std::string(argv[2]) + "/model");
}
