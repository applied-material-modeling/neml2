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
// Build canonical random inputs at batch `b` from the model's declared names +
// base shapes: each input is (b, *base_shape).
std::map<std::string, at::Tensor>
make_inputs(const Model & model, int64_t b)
{
  const auto & names = model.input_names();
  const auto & bases = model.input_base_shapes();
  const auto opts = at::TensorOptions().dtype(model.dtype()).device(model.device());
  std::map<std::string, at::Tensor> inputs;
  for (std::size_t i = 0; i < names.size(); ++i)
  {
    std::vector<int64_t> shape{b};
    shape.insert(shape.end(), bases[i].begin(), bases[i].end());
    inputs.emplace(names[i], at::randn(shape, opts));
  }
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
  const auto tangents = make_inputs(ref, b); // distinct random tangents for jvp
  const auto ref_jvp = ref.jvp(inputs, tangents);
  // Parameter-derivative references (the fixture promotes E + nu and compiles
  // `-d :`, so d(stress)/d{E,nu} graphs exist). Cotangents keyed by output at
  // (b, *out_base) for the adjoint.
  std::map<std::string, at::Tensor> cotangents;
  for (std::size_t i = 0; i < ref.output_names().size(); ++i)
  {
    std::vector<int64_t> shape{b};
    shape.insert(
        shape.end(), ref.output_base_shapes()[i].begin(), ref.output_base_shapes()[i].end());
    cotangents.emplace(ref.output_names()[i],
                       at::randn(shape, at::TensorOptions().dtype(at::kDouble)));
  }
  const auto ref_pjac = ref.param_jacobian(inputs);
  const auto ref_pvjp = ref.param_vjp(inputs, cotangents);

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

    // jacobian (outputs + nested variable-pair blocks)
    auto [jout, j] = disp.jacobian(inputs);
    for (const auto & o : ref.output_names())
    {
      NEML2_CHECK(at::allclose(jout.at(o), std::get<0>(ref_jac).at(o), 1e-8, 1e-10));
      for (const auto & i : ref.input_names())
      {
        // The dispatched block must reassemble to the same shape the
        // single-device reference returns: a batched block is concatenated back
        // to leading dim `b`, while a batch-independent block (this model's
        // constant stiffness Jacobian) is passed through unbatched. Comparing to
        // the reference covers both without hard-coding the batch axis.
        NEML2_CHECK(j.at(o).at(i).sizes() == std::get<1>(ref_jac).at(o).at(i).sizes());
        NEML2_CHECK(at::allclose(j.at(o).at(i), std::get<1>(ref_jac).at(o).at(i), 1e-8, 1e-10));
      }
    }

    // jvp (outputs + directional derivative along `tangents`)
    auto [vout, vdot] = disp.jvp(inputs, tangents);
    for (const auto & name : ref.output_names())
    {
      NEML2_CHECK(at::allclose(vout.at(name), std::get<0>(ref_jvp).at(name), 1e-8, 1e-10));
      NEML2_CHECK(at::allclose(vdot.at(name), std::get<1>(ref_jvp).at(name), 1e-8, 1e-10));
    }

    // param_jacobian: outputs + nested {out: {param: block}}. d(stress)/d{E,nu}
    // is batch-dependent here, so chunked blocks are concatenated back to b.
    auto [pout, P] = disp.param_jacobian(inputs);
    for (const auto & o : ref.output_names())
    {
      NEML2_CHECK(at::allclose(pout.at(o), std::get<0>(ref_pjac).at(o), 1e-8, 1e-10));
      for (const auto & [pname, refblock] : std::get<1>(ref_pjac).at(o))
      {
        NEML2_CHECK(P.at(o).at(pname).sizes() == refblock.sizes());
        NEML2_CHECK(at::allclose(P.at(o).at(pname), refblock, 1e-8, 1e-10));
      }
    }

    // param_vjp: the per-parameter adjoint is summed (not concatenated) across
    // chunks -- each chunk already reduced its own batch slice.
    auto g = disp.param_vjp(inputs, cotangents);
    NEML2_CHECK(g.size() == ref_pvjp.size());
    for (const auto & [pname, refgrad] : ref_pvjp)
    {
      NEML2_CHECK(g.at(pname).sizes() == refgrad.sizes());
      NEML2_CHECK(at::allclose(g.at(pname), refgrad, 1e-8, 1e-10));
    }
  }

  // Batched (per-batch-element) promoted parameter. Set ONE scalar promoted
  // parameter to a distinct value per batch row (the other stays global); forward,
  // param_jacobian, and param_vjp must dispatch chunk-wise (each chunk receives the
  // batched parameter sliced to its rows, never the whole tensor) and reproduce a
  // single-shot reference carrying the same batched parameter. For param_vjp this
  // means the batched parameter's adjoint comes back PER-ELEMENT (concatenated
  // across chunks) while the global parameter's is SUMMED.
  {
    // A scalar promoted parameter (natural base shape empty) -- E or nu.
    std::string sp;
    for (const auto & [q, base] : ref.parameter_base_shapes())
      if (base.empty())
      {
        sp = q;
        break;
      }
    NEML2_CHECK(!sp.empty()); // forward_promoted promotes scalar E + nu

    // Distinct per-row values, deterministic.
    const auto pvals =
        at::linspace(0.8, 1.2, b, at::TensorOptions().dtype(ref.dtype()).device(ref.device())) *
        ref.named_parameters().at(sp);

    // Batched single-shot reference.
    Model bref(meta_path);
    bref.set_parameter(sp, pvals);
    const auto bref_out = bref.forward(inputs);
    const auto bref_pjac = bref.param_jacobian(inputs);
    const auto bref_pvjp = bref.param_vjp(inputs, cotangents);
    // Sanity: a per-row parameter makes the d(out)/d(sp) block vary across the
    // batch (so the test cannot pass with the override ignored), and the batched
    // parameter's adjoint is per-element (B,) while a global parameter's stays
    // scalar -- exactly the shapes the chunk stitch must reproduce.
    {
      const auto & blk = std::get<1>(bref_pjac).at(ref.output_names().front()).at(sp);
      NEML2_CHECK(blk.size(0) == b);
      NEML2_CHECK(!at::allclose(blk.index({0}), blk.index({b - 1})));
      NEML2_CHECK(bref_pvjp.at(sp).dim() == 1 && bref_pvjp.at(sp).size(0) == b); // per-element
      for (const auto & [q, g] : bref_pvjp)
        if (q != sp)
          NEML2_CHECK(g.dim() == 0); // global parameter -> scalar adjoint
    }

    // Sync chunked + async hybrid, including chunk sizes that do not divide b.
    // Returns 0 on success, 1 on a failed check (NEML2_CHECK* `return`s an int,
    // so this helper -- and not a void lambda -- is what they must live in).
    auto check_disp = [&](DispatchedModel & disp) -> int
    {
      disp.set_parameter(sp, pvals);

      auto out = disp.forward(inputs);
      for (const auto & name : ref.output_names())
        NEML2_CHECK(at::allclose(out.at(name), bref_out.at(name), 1e-8, 1e-10));

      auto [pout, P] = disp.param_jacobian(inputs);
      for (const auto & o : ref.output_names())
      {
        NEML2_CHECK(at::allclose(pout.at(o), std::get<0>(bref_pjac).at(o), 1e-8, 1e-10));
        for (const auto & [pname, refblock] : std::get<1>(bref_pjac).at(o))
        {
          NEML2_CHECK(P.at(o).at(pname).sizes() == refblock.sizes());
          NEML2_CHECK(at::allclose(P.at(o).at(pname), refblock, 1e-8, 1e-10));
        }
      }

      // param_vjp: batched parameter's adjoint concatenated per-element, global
      // parameter's summed -- both must equal the batched single-shot reference.
      auto g = disp.param_vjp(inputs, cotangents);
      NEML2_CHECK(g.size() == bref_pvjp.size());
      for (const auto & [pname, refgrad] : bref_pvjp)
      {
        NEML2_CHECK(g.at(pname).sizes() == refgrad.sizes());
        NEML2_CHECK(at::allclose(g.at(pname), refgrad, 1e-8, 1e-10));
      }
      return 0;
    };

    for (std::size_t k : {std::size_t{3}, std::size_t{4}, std::size_t{10}, std::size_t{0}})
    {
      DispatchedModel disp(artifact_root,
                           std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", k}));
      if (check_disp(disp) != 0)
        return 1;
    }
    {
      // Async path: workers build their own per-chunk override concurrently.
      StaticHybridScheduler::Config cfg;
      cfg.devices = {"cpu"};
      cfg.batch_sizes = {3};
      DispatchedModel disp(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));
      if (check_disp(disp) != 0)
        return 1;
    }
  }

  // set_parameter on the dispatched handle updates the master and is broadcast
  // to every device copy on the next call, matching a single-shot reference with
  // the same parameter set (this also exercises aoti::Model::set_parameter via
  // ref2). CPU-only so it is independent of the cuda cross-device path.
  {
    auto scheduler = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 4});
    DispatchedModel disp(artifact_root, scheduler);
    const std::string pname = ref.named_parameters().begin()->first;
    const auto newval = ref.named_parameters().at(pname).clone() * 1.5;
    disp.set_parameter(pname, newval);

    Model ref2(meta_path);
    ref2.set_parameter(pname, newval);
    const auto out = disp.forward(inputs);
    const auto ref2_out = ref2.forward(inputs);
    for (const auto & name : ref.output_names())
      NEML2_CHECK(at::allclose(out.at(name), ref2_out.at(name), 1e-8, 1e-10));
  }

  // Public API surface: the (Model, scheduler) constructor, move semantics, and
  // the trivial accessors -- kept exercised so they don't silently rot.
  {
    DispatchedModel m(std::make_unique<Model>(meta_path),
                      std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 0}));
    NEML2_CHECK(m.input_names() == ref.input_names());
    NEML2_CHECK(m.output_names() == ref.output_names());
    NEML2_CHECK(m.input_base_shapes() == ref.input_base_shapes());
    NEML2_CHECK(m.output_base_shapes() == ref.output_base_shapes());
    NEML2_CHECK(m.dtype() == ref.dtype());
    NEML2_CHECK(!m.named_parameters().empty()); // non-const overload
    NEML2_CHECK(!static_cast<const DispatchedModel &>(m).named_parameters().empty()); // const
    const auto & first = ref.output_names().front();
    NEML2_CHECK(at::allclose(m.forward(inputs).at(first), ref_out.at(first), 1e-8, 1e-10));

    DispatchedModel moved(std::move(m)); // move ctor
    NEML2_CHECK(moved.input_names() == ref.input_names());
    DispatchedModel m2(std::make_unique<Model>(meta_path),
                       std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 0}));
    m2 = std::move(moved); // move assignment
    NEML2_CHECK(at::allclose(m2.forward(inputs).at(first), ref_out.at(first), 1e-8, 1e-10));
  }

  // A foreign (torch) error raised inside Model itself is normalized to a
  // non-recoverable FatalError by its _guarded facade (undefined input tensor).
  {
    std::map<std::string, at::Tensor> undef;
    undef.emplace(ref.input_names().front(), at::Tensor{});
    bool fatal = false;
    try
    {
      (void)ref.forward(undef);
    }
    catch (const FatalError & e)
    {
      fatal = !e.recoverable();
    }
    NEML2_CHECK(fatal);
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
    for (const auto & o : ref.output_names())
      for (const auto & i : ref.input_names())
        NEML2_CHECK(at::allclose(j.at(o).at(i), std::get<1>(ref_jac).at(o).at(i), 1e-8, 1e-10));

    auto [vout, vdot] = disp.jvp(inputs, tangents); // async run_async<Pair> path
    for (const auto & name : ref.output_names())
      NEML2_CHECK(at::allclose(vdot.at(name), std::get<1>(ref_jvp).at(name), 1e-8, 1e-10));
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

    // A foreign (torch) error is normalized to FatalError at the boundary too:
    // an undefined input tensor makes torch throw inside the dispatch, before any
    // neml2 check -- the guard wraps it as a non-recoverable FatalError.
    std::map<std::string, at::Tensor> undef;
    undef.emplace(inputs.begin()->first, at::Tensor{});
    bool fthrew = false, frecoverable = true;
    try
    {
      (void)disp.forward(undef);
    }
    catch (const FatalError & e)
    {
      fthrew = true;
      frecoverable = e.recoverable();
    }
    NEML2_CHECK(fthrew);
    NEML2_CHECK(!frecoverable);

    // Reusable after the failures: a valid call still matches the single shot.
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

    // Batched promoted parameter across the real CPU+GPU hybrid: every chunk --
    // on cpu or cuda -- must receive the parameter sliced (and moved) to its own
    // rows. Compare to a cpu single-shot reference with the same batched value.
    {
      std::string sp;
      for (const auto & [q, base] : ref.parameter_base_shapes())
        if (base.empty())
        {
          sp = q;
          break;
        }
      NEML2_CHECK(!sp.empty());
      const auto pvals = at::linspace(0.8, 1.2, b, at::TensorOptions().dtype(ref.dtype())) *
                         ref.named_parameters().at(sp);

      Model bref(meta_path); // cpu single-shot, batched
      bref.set_parameter(sp, pvals);
      const auto bref_out = bref.forward(inputs);
      const auto bref_pjac = bref.param_jacobian(inputs);

      StaticHybridScheduler::Config cfg;
      cfg.devices = {"cpu", "cuda"};
      cfg.batch_sizes = {3, 4};
      DispatchedModel disp(artifact_root, std::make_shared<StaticHybridScheduler>(cfg));
      disp.set_parameter(sp, pvals);

      auto out = disp.forward(inputs);
      for (const auto & name : ref.output_names())
      {
        NEML2_CHECK(out.at(name).device().is_cpu());
        NEML2_CHECK(at::allclose(out.at(name), bref_out.at(name), 1e-6, 1e-8));
      }
      auto [pout, P] = disp.param_jacobian(inputs);
      for (const auto & o : ref.output_names())
        for (const auto & [pname, refblock] : std::get<1>(bref_pjac).at(o))
          NEML2_CHECK(at::allclose(P.at(o).at(pname), refblock, 1e-6, 1e-8));
    }
  }

  return 0;
}
