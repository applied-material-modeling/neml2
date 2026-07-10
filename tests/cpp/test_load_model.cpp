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

// Exercises the C++ load_model(stub, name[, scheduler]) factory against the
// standalone-stub fixture: <dir>/model_aoti.i + <dir>/model/<device>/...
// (compiled by the dispatcher_fixture_compile ctest fixture). argv[1] is the
// fixture (collection) dir.

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/Model.h"
#include "neml2/csrc/dispatchers/SimpleScheduler.h"
#include "neml2/csrc/dispatchers/factory.h"

#include "test_util.h"

using namespace neml2::aoti;

namespace
{
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

bool
matches(const std::map<std::string, at::Tensor> & got,
        const std::map<std::string, at::Tensor> & ref,
        double rtol,
        double atol)
{
  for (const auto & [name, t] : ref)
  {
    auto it = got.find(name);
    if (it == got.end() || !at::allclose(it->second, t, rtol, atol))
      return false;
  }
  return true;
}
} // namespace

int
main(int argc, char ** argv)
{
  NEML2_CHECK(argc >= 2); // argv[1] = the fixture (collection) dir
  const std::string fixture_dir = argv[1];
  const std::string stub = fixture_dir + "/model_aoti.i";
  const std::string artifact_root = fixture_dir + "/model";

  at::manual_seed(0);

  // Single-shot reference straight off the cpu artifact root (schema v10: the
  // Model ctor takes the artifact folder + device + dtype and resolves
  // metadata.json + the <device>/<dtype>/ binaries itself).
  Model ref(artifact_root, at::kCPU, at::kDouble);
  const int64_t b = 10;
  const auto inputs = make_inputs(ref, b);
  const auto ref_out = ref.forward(inputs);

  // No scheduler -> pass-through on cpu; must match the single shot.
  {
    auto m = load_model(stub, "model");
    NEML2_CHECK(m.device().is_cpu());
    NEML2_CHECK(m.input_names() == ref.input_names());
    NEML2_CHECK(matches(m.forward(inputs), ref_out, 1e-8, 1e-10));
  }

  // Explicit cpu scheduler with chunking (3 does not divide 10).
  {
    auto sched = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", 3});
    auto m = load_model(stub, "model", sched);
    NEML2_CHECK(matches(m.forward(inputs), ref_out, 1e-8, 1e-10));
  }

  // Unknown model name -> clean error.
  NEML2_CHECK_THROWS(load_model(stub, "does_not_exist"));

  // --- factory.cpp branches via hand-written stubs (no extra artifact needed) ---
  // Schema v10: the stub carries only type + artifact_path (pointing at the
  // artifact root); solver config lives in the artifact's metadata.json, so
  // factory::load_model no longer parses a [Solvers] block or a `solver` field.
  namespace fs = std::filesystem;
  // Absolute, since the stubs below live in a temp dir: a relative artifact_path
  // would resolve against that temp dir, not the fixture.
  const std::string artifact = fs::absolute(fs::path(fixture_dir) / "model").string();
  const auto tmp = fs::temp_directory_path() / "neml2_load_model_test";
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto write_stub = [&](const std::string & name, const std::string & body)
  {
    const auto p = (tmp / name).string();
    std::ofstream(p) << body;
    return p;
  };

  // A minimal stub (type + absolute artifact_path) loads through factory::load_model.
  {
    const auto s =
        write_stub("plain.i",
                   "[Models]\n  [model]\n    type = AOTIModel\n    artifact_path = " + artifact +
                       "\n  []\n[]\n");
    NEML2_CHECK(matches(load_model(s, "model").forward(inputs), ref_out, 1e-8, 1e-10));
  }
  // Relative artifact_path -> resolved against the stub's own directory, so the
  // stub must sit next to the `model` folder (i.e. in fixture_dir).
  {
    const auto s = (fs::path(fixture_dir) / "rel_stub.i").string();
    std::ofstream(s)
        << "[Models]\n  [model]\n    type = AOTIModel\n    artifact_path = model\n  []\n[]\n";
    NEML2_CHECK(matches(load_model(s, "model").forward(inputs), ref_out, 1e-8, 1e-10));
    fs::remove(s);
  }
  // Error branches: parse failure, wrong type, missing artifact_path.
  NEML2_CHECK_THROWS(load_model("/no/such/stub.i", "model"));
  NEML2_CHECK_THROWS(load_model(
      write_stub("wrongtype.i", "[Models]\n  [model]\n    type = NotAOTI\n  []\n[]\n"), "model"));
  NEML2_CHECK_THROWS(load_model(
      write_stub("noart.i", "[Models]\n  [model]\n    type = AOTIModel\n  []\n[]\n"), "model"));
  fs::remove_all(tmp);

  // Cross-device: dispatch cpu inputs onto the GPU when a cuda artifact exists.
  const std::string cuda_leaf = fixture_dir + "/model/cuda/float64";
  if (at::hasCUDA() && std::filesystem::exists(cuda_leaf))
  {
    auto sched = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cuda", 4});
    auto m = load_model(stub, "model", sched);
    NEML2_CHECK(m.device().is_cuda());
    auto out = m.forward(inputs); // inputs live on cpu
    for (const auto & name : ref.output_names())
      NEML2_CHECK(out.at(name).device().is_cpu()); // returned on the input device
    NEML2_CHECK(matches(out, ref_out, 1e-6, 1e-8));
  }

  return 0;
}
