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

// End-to-end: a boundary-renamed artifact must report the RENAMED names on both
// the C++ `Model` facade and the `DispatchedModel`, accept / return them on every
// op, and keep dispatch parity (chunked == single shot). The fixture compiles
// forward_promoted (LinearIsotropicElasticity, E + nu promoted) with
// `--rename-input strain:eps --rename-output stress:sig
//  --rename-parameter model.E:youngs --rename-parameter model.nu:poisson`, so the
// authored names must NOT appear at the boundary while the values are unchanged.

#include <filesystem>
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
  NEML2_CHECK(argc >=
              2); // argv[1] = the renamed fixture artifact root (holds metadata.json + cpu/)
  const std::string artifact_root = argv[1];

  at::manual_seed(0);

  Model ref(artifact_root, at::kCPU, at::kDouble);

  // The boundary reports the RENAMED names, never the authored ones.
  NEML2_CHECK(ref.input_names() == std::vector<std::string>{"eps"});
  NEML2_CHECK(ref.output_names() == std::vector<std::string>{"sig"});
  const auto & params = ref.named_parameters();
  NEML2_CHECK(params.count("youngs") == 1 && params.count("poisson") == 1);
  NEML2_CHECK(params.count("model.E") == 0 && params.count("model.nu") == 0);
  const auto & pbases = ref.parameter_base_shapes();
  NEML2_CHECK(pbases.count("youngs") == 1 && pbases.count("poisson") == 1);

  const int64_t b = 10;
  const auto inputs = make_inputs(ref, b); // keyed by "eps"
  NEML2_CHECK(inputs.count("eps") == 1);

  const auto ref_out = ref.forward(inputs);
  NEML2_CHECK(ref_out.count("sig") == 1 && ref_out.count("stress") == 0);

  const auto ref_jac = ref.jacobian(inputs);
  NEML2_CHECK(std::get<1>(ref_jac).at("sig").count("eps") == 1);
  const auto ref_pjac = ref.param_jacobian(inputs);
  NEML2_CHECK(std::get<1>(ref_pjac).at("sig").count("youngs") == 1);

  // Dispatch parity across chunk sizes -- the dispatcher stitches by the renamed
  // keys, so its results match the single-shot reference bit-for-bit.
  for (std::size_t k : {std::size_t{3}, std::size_t{4}, std::size_t{0}})
  {
    auto scheduler = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cpu", k});
    DispatchedModel disp(artifact_root, scheduler);

    NEML2_CHECK(disp.input_names() == std::vector<std::string>{"eps"});
    NEML2_CHECK(disp.output_names() == std::vector<std::string>{"sig"});
    NEML2_CHECK(disp.named_parameters().count("youngs") == 1);
    NEML2_CHECK(disp.named_parameters().count("model.E") == 0);

    auto out = disp.forward(inputs);
    NEML2_CHECK(out.count("sig") == 1);
    NEML2_CHECK(at::allclose(out.at("sig"), ref_out.at("sig"), 1e-8, 1e-10));

    auto [jout, j] = disp.jacobian(inputs);
    NEML2_CHECK(
        at::allclose(j.at("sig").at("eps"), std::get<1>(ref_jac).at("sig").at("eps"), 1e-8, 1e-10));

    auto [pout, P] = disp.param_jacobian(inputs);
    NEML2_CHECK(at::allclose(
        P.at("sig").at("youngs"), std::get<1>(ref_pjac).at("sig").at("youngs"), 1e-8, 1e-10));
  }

  // set_parameter by the BOUNDARY name flows into the value graph: perturbing
  // "youngs" (the renamed model.E) changes the forward output.
  Model mutated(artifact_root, at::kCPU, at::kDouble);
  const auto base_out = mutated.forward(inputs).at("sig").clone();
  mutated.set_parameter("youngs", mutated.named_parameters().at("youngs") * 1.5);
  const auto new_out = mutated.forward(inputs).at("sig");
  NEML2_CHECK(!at::allclose(base_out, new_out));

  return 0;
}
