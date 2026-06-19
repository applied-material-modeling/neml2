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

// End-to-end: dispatching a workload in chunks must reproduce a single-shot
// Model run, bit-for-bit (the leaf is per-batch-element independent, so chunk
// boundaries cannot change the result). The artifact is compiled by the
// `dispatcher_fixture_compile` ctest fixture (forward_single == one
// LinearIsotropicElasticity leaf, cpu) into <fixture_dir>/cpu/.

#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/DispatchedModel.h"
#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/aoti/SimpleScheduler.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
// Build shape-correct random inputs at batch `b` from the model's declared
// names + flat var sizes. The fixture leaf has a single 1-D-base input
// (strain, var_size 6), so (b, var_size) matches the traced (dyn, base) rank.
std::map<std::string, at::Tensor>
make_inputs(const Model & model, int64_t b)
{
  const auto & names = model.input_names();
  const auto & sizes = model.input_sizes();
  const auto opts = at::TensorOptions().dtype(model.dtype()).device(model.device());
  std::map<std::string, at::Tensor> inputs;
  for (std::size_t i = 0; i < names.size(); ++i)
    inputs.emplace(names[i], at::randn({b, sizes[i]}, opts));
  return inputs;
}
} // namespace

int
main(int argc, char ** argv)
{
  NEML2_CHECK(argc >= 2); // argv[1] = the fixture artifact root (holds cpu/ subfolder)
  const std::string artifact_root = argv[1];
  const std::string meta_path = artifact_root + "/cpu/model_meta.json";

  at::manual_seed(0);

  // Single-shot reference.
  Model ref(meta_path);
  const int64_t b = 10;
  const auto inputs = make_inputs(ref, b);
  const auto ref_out = ref.forward(inputs);
  const auto ref_jac = ref.jacobian(inputs);

  // Chunked dispatch must match for every chunk size, including ones that do
  // not evenly divide the batch, the exact-batch case, and the no-chunk
  // sentinel (0).
  for (std::size_t k : {std::size_t{3}, std::size_t{4}, std::size_t{10}, std::size_t{0}})
  {
    auto scheduler = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", k});
    DispatchedModel disp(artifact_root, scheduler);

    NEML2_CHECK(disp.input_names() == ref.input_names());
    NEML2_CHECK(disp.device().is_cpu());

    // forward
    auto out = disp.forward(inputs);
    for (const auto & name : ref.output_names())
    {
      NEML2_CHECK(out.count(name) == 1);
      NEML2_CHECK(out.at(name).size(0) == b);
      NEML2_CHECK(at::allclose(out.at(name), ref_out.at(name), /*rtol=*/1e-8, /*atol=*/1e-10));
    }

    // jacobian (outputs + assembled J)
    auto [jout, j] = disp.jacobian(inputs);
    NEML2_CHECK(j.size(0) == b);
    NEML2_CHECK(at::allclose(j, std::get<1>(ref_jac), /*rtol=*/1e-8, /*atol=*/1e-10));
    for (const auto & name : ref.output_names())
      NEML2_CHECK(at::allclose(jout.at(name), std::get<0>(ref_jac).at(name), 1e-8, 1e-10));
  }

  // Cross-device dispatch: when a GPU and a cuda artifact are both present,
  // dispatch the CPU-resident inputs onto the GPU and verify the results --
  // returned on the input (CPU) device -- match the CPU single-shot reference.
  // The cpu-only loop above cannot reach this path (there in_device ==
  // compute_device, so no transfer happens). Skipped on cpu-only machines.
  const std::string cuda_meta = artifact_root + "/cuda/model_meta.json";
  if (at::hasCUDA() && std::filesystem::exists(cuda_meta))
  {
    auto scheduler = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cuda", 4});
    DispatchedModel disp(artifact_root, scheduler);
    NEML2_CHECK(disp.device().is_cuda());

    auto out = disp.forward(inputs); // inputs live on cpu
    for (const auto & name : ref.output_names())
    {
      NEML2_CHECK(out.at(name).device().is_cpu()); // returned on the input device
      NEML2_CHECK(at::allclose(out.at(name), ref_out.at(name), /*rtol=*/1e-6, /*atol=*/1e-8));
    }
  }

  return 0;
}
