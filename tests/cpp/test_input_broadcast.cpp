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

// Exercises the cpp-aoti runtime with a batch-INDEPENDENT input alongside a
// batched one -- the shape MOOSE produces for a scalar TIME force. The dispatcher
// fixtures (test_load_model / test_dispatcher) and test_custom_ops all feed
// uniformly-batched inputs, so the runtime's input-broadcast path was uncovered;
// the cpp-aoti Model must lift a batch-independent input up to the common call
// batch the same way the typed (eager / py-aoti shim) routes do.
//
// Model: ScalarLinearCombination C = A + B + offset, with A batched and B a 0-dim
// scalar. forward() broadcasts inside the graph, so the *jacobian* -- whose block
// assembly cats per-input columns -- is the discriminating call: without the
// broadcast it raises "Tensors must have same number of dimensions: got 2 and 1".
// argv[1] is the fixture (collection) dir.

#include <map>
#include <string>

#include <ATen/ATen.h>

#include "neml2/csrc/aoti/Model.h"

#include "test_util.h"

using namespace neml2::aoti;

int
main(int argc, char ** argv)
{
  NEML2_CHECK(argc >= 2); // argv[1] = the fixture (collection) dir
  const std::string artifact_root = std::string(argv[1]) + "/model";

  Model m(artifact_root, at::kCPU, at::kDouble);
  const int64_t b = 6;
  const double offset = 2.0; // matches batch_broadcast_model.i
  const auto opts = at::TensorOptions().dtype(m.dtype()).device(m.device());

  const auto A = at::arange(b, opts);     // batched      -> (b,)
  const auto B = at::full({}, 5.0, opts); // batch-indep. -> 0-dim scalar
  const std::map<std::string, at::Tensor> ins{{"A", A}, {"B", B}};

  // jacobian() drives the block-assembly cat; without the input broadcast the
  // 0-dim B collides with the batched A and this throws. The value half of the
  // result also confirms B was lifted to the batch: C = A + B + offset.
  const auto outputs = m.jacobian(ins).first;
  NEML2_CHECK(at::allclose(outputs.at("C"), A + 5.0 + offset));

  return 0;
}
