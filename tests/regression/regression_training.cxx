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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "utils.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/TensorValue.h"

using namespace neml2;

TEST_CASE("training")
{
  // This regression test loads a NEML2 model as if it is being used as an external library/package.
  // A parameter gradient is requested before the model is called twice, after which backward() is
  // called on the objective function to propagate the gradient onto the parameter.
  auto model = load_model("regression_training.i", "model");

  // Batch shape
  Size nbatch = 2;

  // Request parameter gradient
  auto & p = model->get_parameter("yield_sy");
  p = Scalar::create(5, default_tensor_options().requires_grad(true));

  // Variables
  VariableName strain(FORCES, "E");
  VariableName time(FORCES, "t");
  VariableName stress(STATE, "S");
  VariableName ep(STATE, "internal", "ep");

  // Evaluate the model for the first time
  ValueMap x1;
  x1[strain.old()] = SR2::fill(0.0, 0.0, 0.01, -0.01, -0.01, 0.02).batch_expand(nbatch);
  x1[strain] = SR2::fill(0.01, 0.01, 0.01, -0.02, -0.03, 0.04).batch_expand(nbatch);
  x1[time] = Scalar::full(1.0).batch_expand(nbatch);
  x1[stress] = SR2::fill(100.0, 100.0, 200.0, -50.0, -150.0, 50.0).batch_expand(nbatch);
  x1[ep] = Scalar::full(0.001).batch_expand(nbatch);
  const auto r1 = model->value(x1);

  // Evaluate the model for the second time
  ValueMap x2;
  x2[strain.old()] = x1[strain];
  x2[time.old()] = x1[time];
  x2[stress.old()] = r1.at(stress.remount(RESIDUAL)) * 1e-2;
  x2[ep.old()] = r1.at(ep.remount(RESIDUAL)) * 1e-2;
  x2[strain] = SR2::fill(0.02, 0.02, 0.03, -0.02, -0.01, 0.01).batch_expand(nbatch);
  x2[time] = Scalar::full(5.0).batch_expand(nbatch);
  x2[stress] = SR2::fill(100.0, 100.0, 200.0, -50.0, -150.0, 50.0).batch_expand(nbatch);
  x2[ep] = Scalar::full(0.001).batch_expand(nbatch);
  const auto r2 = model->value(x2);

  // Evaluate the objective function
  const auto s2 = r2.at(stress.remount(RESIDUAL)) * 1e-2;
  const auto ep2 = r2.at(ep.remount(RESIDUAL)) * 1e-2;
  const auto f = at::norm(at::cat({s2.base_flatten(), ep2.base_flatten()}, -1));

  // Check the parameter gradient
  f.backward();
  REQUIRE(Tensor(p).grad().item<double>() == Catch::Approx(-172.543));
}
