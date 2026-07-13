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

// Exercises the cpp-aoti runtime with a model that uses a NEML2 custom op that
// torch.export deliberately preserves into the artifact -- neml2::opaque_pow,
// the Inductor fusion barrier from neml2/types/functions.py. The other AOTI
// dispatcher fixtures use LinearIsotropicElasticity (no custom ops), so nothing
// covered the path where the Python-free libneml2 must register the op itself:
// the Python package's registration is absent in a pure-C++ host, and the
// dispatcher otherwise throws "Could not find schema for neml2::opaque_pow"
// when the artifact loads. The fixture compiles PerzynaPlasticFlowRate, whose
// flow_rate = (yield_function / reference_stress)^exponent routes through
// opaque_pow. argv[1] is the fixture (collection) dir.

#include <filesystem>
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
// reference_stress=150, exponent=6, yield_function=50:
// flow_rate = (50/150)^6 = (1/3)^6 = 1/729 -- the PerzynaPlasticFlowRate
// ModelUnitTest gold. A value-level check (not just "load succeeds") confirms
// the C++ opaque_pow kernel computes the right thing, not merely that a schema
// is present.
constexpr double kYieldFunction = 50.0;
constexpr double kExpectedFlowRate = 0.0013717421124828527;

std::map<std::string, at::Tensor>
make_inputs(const Model & model, int64_t b)
{
  std::vector<int64_t> shape{b};
  const auto & base = model.input_base_shapes()[0];
  shape.insert(shape.end(), base.begin(), base.end());
  const auto opts = at::TensorOptions().dtype(model.dtype()).device(model.device());
  return {{"yield_function", at::full(shape, kYieldFunction, opts)}};
}

// forward() resolves neml2::opaque_pow through the dispatcher; without the C++
// registration in libneml2 this throws before returning. Templated because the
// load_model factory returns a DispatchedModel while the direct artifact is a
// Model -- both expose forward().
template <typename ModelT>
bool
flow_rate_ok(const ModelT & model,
             const std::map<std::string, at::Tensor> & inputs,
             double rtol,
             double atol)
{
  const auto out = model.forward(inputs);
  const auto & fr = out.at("flow_rate");
  return at::allclose(fr, at::full_like(fr, kExpectedFlowRate), rtol, atol);
}
} // namespace

int
main(int argc, char ** argv)
{
  NEML2_CHECK(argc >= 2); // argv[1] = the fixture (collection) dir
  const std::string fixture_dir = argv[1];
  const std::string stub = fixture_dir + "/model_aoti.i";
  const std::string artifact_root = fixture_dir + "/model";

  const int64_t b = 8;

  // Direct cpu artifact.
  Model ref(artifact_root, at::kCPU, at::kDouble);
  NEML2_CHECK(ref.input_names().size() == 1);
  NEML2_CHECK(ref.input_names()[0] == "yield_function");
  const auto inputs = make_inputs(ref, b);
  NEML2_CHECK(flow_rate_ok(ref, inputs, 1e-10, 1e-12));

  // Through the load_model(stub, name) factory.
  {
    auto m = load_model(stub, "model");
    NEML2_CHECK(m.device().is_cpu());
    NEML2_CHECK(flow_rate_ok(m, inputs, 1e-10, 1e-12));
  }

  // cuda: dispatch the cpu inputs onto the GPU when a cuda artifact exists, so
  // the op resolves on the cuda backend too.
  const std::string cuda_leaf = fixture_dir + "/model/cuda/float64";
  if (at::hasCUDA() && std::filesystem::exists(cuda_leaf))
  {
    auto sched = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cuda", 4});
    auto m = load_model(stub, "model", sched);
    NEML2_CHECK(m.device().is_cuda());
    NEML2_CHECK(flow_rate_ok(m, inputs, 1e-8, 1e-10));
  }

  return 0;
}
