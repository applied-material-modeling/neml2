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

// Exercises the embedded-Python eager runtime: neml2::eager::load_model loads a
// model from the ORIGINAL .i (no AOTI compile, no stub) by embedding a CPython
// interpreter, and forward() evaluates it. argv[1] is the path to
// tests/aoti/forward_single/model.i (LinearIsotropicElasticity, strain->stress).

#include <map>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/Exception.h"
#include "neml2/csrc/eager/Model.h"
#include "neml2/csrc/eager/load_model.h"

#include "test_util.h"

using namespace neml2::eager;

namespace
{
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
  NEML2_CHECK(argc >= 2); // argv[1] = the original .i input file
  const std::string input_file = argv[1];

  at::manual_seed(0);

  auto m = load_model(input_file, "model");

  // Metadata parity with the model declared in the .i (LinearIsotropicElasticity:
  // SR2 strain -> SR2 stress, base size 6). These are the same names/sizes the
  // AOTI metadata path reports for this model, so eager is a drop-in.
  NEML2_CHECK(m.input_names() == std::vector<std::string>{"strain"});
  NEML2_CHECK(m.output_names() == std::vector<std::string>{"stress"});
  NEML2_CHECK(m.input_sizes() == std::vector<int>{6});
  NEML2_CHECK(m.output_sizes() == std::vector<int>{6});
  NEML2_CHECK(m.device().is_cpu());
  NEML2_CHECK(m.dtype() == at::kDouble);

  const int64_t b = 8;
  const auto inputs = make_inputs(m, b);

  const auto out = m.forward(inputs);
  NEML2_CHECK(out.count("stress") == 1);
  NEML2_CHECK(out.at("stress").dim() == 2);
  NEML2_CHECK(out.at("stress").size(0) == b);
  NEML2_CHECK(out.at("stress").size(1) == 6);
  NEML2_CHECK(out.at("stress").device().is_cpu());
  NEML2_CHECK(out.at("stress").dtype() == at::kDouble);
  NEML2_CHECK(at::isfinite(out.at("stress")).all().item<bool>());

  // Deterministic: a second call on the same inputs gives the same result.
  const auto out2 = m.forward(inputs);
  NEML2_CHECK(at::allclose(out.at("stress"), out2.at("stress")));

  // jacobian: assembled (B, sum(out), sum(in)) = (B, 6, 6); value half matches
  // forward. (Value-for-value parity vs autograd is checked in tests/unit.)
  {
    const auto [jout, J] = m.jacobian(inputs);
    NEML2_CHECK(J.dim() == 3);
    NEML2_CHECK(J.size(0) == b);
    NEML2_CHECK(J.size(1) == 6);
    NEML2_CHECK(J.size(2) == 6);
    NEML2_CHECK(at::isfinite(J).all().item<bool>());
    NEML2_CHECK(at::allclose(jout.at("stress"), out.at("stress")));

    // jvp(inputs, tangents) == J @ tangents, and the value half matches forward.
    std::map<std::string, at::Tensor> tang;
    tang.emplace("strain", at::randn({b, 6}, at::TensorOptions().dtype(m.dtype())));
    const auto [vout, jvp] = m.jvp(inputs, tang);
    NEML2_CHECK(at::allclose(vout.at("stress"), out.at("stress")));
    NEML2_CHECK(jvp.at("stress").size(0) == b);
    NEML2_CHECK(jvp.at("stress").size(1) == 6);
    const auto jv = at::einsum("bij,bj->bi", {J, tang.at("strain")});
    NEML2_CHECK(at::allclose(jvp.at("stress"), jv));

    // A missing tangent defaults to zero -> zero directional derivative.
    const auto [_, jvp0] = m.jvp(inputs, {});
    NEML2_CHECK(at::allclose(jvp0.at("stress"), at::zeros_like(jvp0.at("stress"))));
  }

  // A second handle shares the embedded interpreter and agrees value-for-value.
  {
    auto m2 = load_model(input_file, "model");
    NEML2_CHECK(at::allclose(m2.forward(inputs).at("stress"), out.at("stress")));
  }

  // Explicit device override (the 3-arg ctor): pins the model to a device. "cpu"
  // keeps the test runnable everywhere; result must match the default-device run.
  {
    Model m_cpu(input_file, "model", at::Device(at::kCPU));
    NEML2_CHECK(m_cpu.device().is_cpu());
    NEML2_CHECK(at::allclose(m_cpu.forward(inputs).at("stress"), out.at("stress")));
  }

  // Error paths -- unknown model name, missing file.
  NEML2_CHECK_THROWS(load_model(input_file, "does_not_exist"));
  NEML2_CHECK_THROWS(load_model("/no/such/file.i", "model"));

  // Wrong-dtype input: the Python boundary check raises; the eager runtime
  // normalizes it to neml2::aoti::FatalError across the .so boundary.
  {
    auto bad = inputs;
    bad.at("strain") = bad.at("strain").to(at::kFloat);
    NEML2_CHECK_THROWS(m.forward(bad));
  }

  // Missing input -> raises.
  {
    const std::map<std::string, at::Tensor> empty;
    NEML2_CHECK_THROWS(m.forward(empty));
  }

  // The normalized exception is the aoti taxonomy (FatalError), catchable across
  // the libneml2.so / libneml2_eager.so boundary via the shared exported typeinfo.
  {
    bool got_fatal = false;
    try
    {
      load_model(input_file, "does_not_exist");
    }
    catch (const neml2::aoti::FatalError &)
    {
      got_fatal = true;
    }
    catch (...)
    {
    }
    NEML2_CHECK(got_fatal);
  }

  // #4: device-override to CUDA, gated on availability so the cpu cell skips it
  // and the gpu_runner cell exercises the py::cast(*device_override) -> .to(cuda)
  // path end to end (forward + jacobian land on the GPU).
  if (at::hasCUDA())
  {
    Model m_cuda(input_file, "model", at::Device(at::kCUDA));
    NEML2_CHECK(m_cuda.device().is_cuda());
    const auto cuda_inputs = make_inputs(m_cuda, b);
    const auto cout = m_cuda.forward(cuda_inputs);
    NEML2_CHECK(cout.at("stress").device().is_cuda());
    NEML2_CHECK(at::isfinite(cout.at("stress")).all().item<bool>());
    const auto [jc, Jc] = m_cuda.jacobian(cuda_inputs);
    (void)jc;
    NEML2_CHECK(Jc.device().is_cuda());
  }

  // #3: a non-converging implicit model surfaces a *recoverable* ConvergenceError
  // across the .so boundary. It originates as neml2::aoti::ConvergenceError in
  // libneml2.so's Newton, round-trips through the embedded interpreter as the
  // registered neml2.aoti._aoti.ConvergenceError, and the eager runtime re-raises
  // it as the typed C++ exception (NOT a plain FatalError). argv[2] is the
  // max_its=0 fixture (tests/cpp/fixtures/implicit_diverge.i).
  if (argc >= 3)
  {
    auto md = load_model(argv[2], "model");
    const auto bad_inputs = make_inputs(md, b);
    bool got_conv = false;
    try
    {
      md.forward(bad_inputs);
    }
    catch (const neml2::aoti::ConvergenceError & e)
    {
      got_conv = true;
      NEML2_CHECK(e.recoverable()); // recoverable: a host can cut dt and retry
    }
    NEML2_CHECK(got_conv);
  }

  return 0;
}
