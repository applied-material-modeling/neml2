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

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/dispatchers/DispatchedModel.h"
#include "neml2/csrc/dispatchers/SimpleScheduler.h"
#include "neml2/csrc/dispatchers/StaticHybridScheduler.h"

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

  // Async correctness: a single-CPU StaticHybridScheduler drives the
  // thread-per-device pool (schedule -> dispatch -> worker -> reduce) with one
  // worker; the order-preserving result must equal the synchronous single shot.
  {
    StaticHybridScheduler::Config cfg;
    cfg.devices = {"cpu"};
    cfg.batch_sizes = {3}; // chunks of 3 over b=10
    DispatchedModel disp(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));

    auto out = disp.forward(inputs);
    for (const auto & name : ref.output_names())
      NEML2_CHECK(at::allclose(out.at(name), ref_out.at(name), 1e-8, 1e-10));

    auto [jout, j] = disp.jacobian(inputs);
    NEML2_CHECK(at::allclose(j, std::get<1>(ref_jac), 1e-8, 1e-10));
  }

  // Async error propagation: a chunk that throws inside a worker thread must not
  // call std::terminate, must not deadlock wait_for_completion, and must surface
  // the exception on the calling thread. Trigger it by supplying the input under
  // a bogus name: the map is non-empty (so batch-size inference + slicing run on
  // the main thread), but the model's "missing required input" assertion fires
  // once the *worker* runs the model -- a non-recoverable FatalError. (AOTI
  // kernels trust their static shapes, so a wrong-shaped tensor is read by
  // stride rather than rejected; a missing input is the reliable trigger.) The
  // same handle must stay usable afterwards, proving the scheduler load drained
  // and the worker pool is still alive (a stranded load would hang this test).
  {
    StaticHybridScheduler::Config cfg;
    cfg.devices = {"cpu"};
    cfg.batch_sizes = {3}; // several chunks over b=10
    DispatchedModel disp(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));

    std::map<std::string, at::Tensor> bad;
    bad.emplace("__nonexistent_input__", inputs.begin()->second);

    // Catch the consumer-facing base type: the failure may surface as a lone
    // FatalError or (if several chunks were in flight) an AggregateError, but
    // either way it must be non-recoverable -- a missing input is a hard stop.
    bool threw = false;
    bool recoverable = true;
    try
    {
      (void)disp.forward(bad);
    }
    catch (const neml2::aoti::Exception & e)
    {
      threw = true;
      recoverable = e.recoverable();
    }
    NEML2_CHECK(threw);        // surfaced, not terminate / hang
    NEML2_CHECK(!recoverable); // a missing input is a hard stop, not a retry

    // Reusable after the failure: a valid call still matches the single shot.
    auto out = disp.forward(inputs);
    for (const auto & name : ref.output_names())
      NEML2_CHECK(at::allclose(out.at(name), ref_out.at(name), 1e-8, 1e-10));
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

    // Real CPU+GPU hybrid: concurrent dispatch across both devices must still
    // reproduce the single shot (results gathered back on the CPU input device).
    {
      StaticHybridScheduler::Config cfg;
      cfg.devices = {"cpu", "cuda"};
      cfg.batch_sizes = {3, 4};
      DispatchedModel disp_h(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));
      auto out_h = disp_h.forward(inputs);
      for (const auto & name : ref.output_names())
      {
        NEML2_CHECK(out_h.at(name).device().is_cpu());
        NEML2_CHECK(at::allclose(out_h.at(name), ref_out.at(name), 1e-6, 1e-8));
      }
    }

    // Write-through promoted parameters: mutating the master must reach the GPU
    // copy too. Perturb a parameter on a fresh single-shot reference and on the
    // hybrid handle by the same amount; the hybrid result (whose chunks split
    // across cpu + cuda) must match -- which it only can if the cuda copy saw
    // the mutation.
    {
      Model ref_mut(meta_path);
      auto & rp = ref_mut.named_parameters();
      NEML2_CHECK(!rp.empty()); // forward_promoted exposes E + nu
      const std::string pkey = rp.begin()->first;
      rp.at(pkey).mul_(1.1);
      const auto ref_mut_out = ref_mut.forward(inputs);
      // Sanity: the parameter actually moves the output.
      const auto & first = ref.output_names().front();
      NEML2_CHECK(!at::allclose(ref_mut_out.at(first), ref_out.at(first)));

      StaticHybridScheduler::Config cfg;
      cfg.devices = {"cpu", "cuda"};
      cfg.batch_sizes = {3, 4};
      DispatchedModel disp_w(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));
      disp_w.named_parameters().at(pkey).mul_(1.1); // master -> broadcast before dispatch
      auto out_w = disp_w.forward(inputs);
      for (const auto & name : ref.output_names())
        NEML2_CHECK(at::allclose(out_w.at(name), ref_mut_out.at(name), 1e-6, 1e-8));
    }
  }

  return 0;
}
