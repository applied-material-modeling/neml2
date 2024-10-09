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

using namespace neml2;

TEST_CASE("training")
{
  // This regression test loads a NEML2 model as if it is being used as an external library/package.
  // A parameter gradient is requested before the model is called twice, after which backward() is
  // called on the objective function to propagate the gradient onto the parameter.
  auto & model = reload_model("regression/regression_training.i", "model");

  // Reinitialize the model to have the correct batch shape and derivative order
  Size nbatch = 2;

  // Request parameter gradient
  auto & p = model.named_parameters()["yield.sy"];
  p = Scalar(5, default_tensor_options().requires_grad(true));

  // Initial state
  auto force0 = torch::tensor({0.0, 0.0, 0.01, -0.01, -0.01, 0.02, 0.0}, default_tensor_options())
                    .expand({nbatch, 7});
  auto state0 = torch::zeros({nbatch, 7}, default_tensor_options());

  // Evaluate the model for the first time
  auto force1 = torch::tensor({0.01, 0.01, 0.01, -0.02, -0.03, 0.04, 1.0}, default_tensor_options())
                    .expand({nbatch, 7});
  auto state1 =
      torch::tensor({100.0, 100.0, 200.0, -50.0, -150.0, 50.0, 0.001}, default_tensor_options())
          .expand({nbatch, 7});
  auto x1 = torch::cat({force0, force1, state0, state1}, -1);
  auto y1 = model.value(LabeledVector(Tensor(x1, 1), {&model.input_axis()}));
  state1 = torch::Tensor(y1) * 1e-2;

  // Evaluate the model for the second time
  auto force2 = torch::tensor({0.02, 0.02, 0.03, -0.02, -0.01, 0.01, 5.0}, default_tensor_options())
                    .expand({nbatch, 7});
  auto state2 =
      torch::tensor({100.0, 100.0, 200.0, -50.0, -150.0, 50.0, 0.001}, default_tensor_options())
          .expand({nbatch, 7});
  auto x2 = torch::cat({force1, force2, state1, state2}, -1);
  auto y2 = model.value(LabeledVector(Tensor(x2, 1), {&model.input_axis()}));
  state2 = torch::Tensor(y2) * 1e-2;

  // Evaluate the objective function
  auto f = torch::norm(state2);

  // Check the parameter gradient
  f.backward();
  REQUIRE(Tensor(p).grad().item<Real>() == Catch::Approx(-127.5426));
}
