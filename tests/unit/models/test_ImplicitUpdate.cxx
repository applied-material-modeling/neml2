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

#include "neml2/neml2.h"
#include "neml2/models/ImplicitUpdate.h"
#include "neml2/tensors/Scalar.h"

using namespace neml2;

TEST_CASE("ImplicitUpdate", "[models]")
{
  auto model0 = load_model("models/ImplicitUpdate.i", "model");
  auto model = std::dynamic_pointer_cast<ImplicitUpdate>(model0);
  ValueMap in = {{VariableName(OLD_STATE, "foo"), Scalar::full(0.0)},
                 {VariableName(OLD_STATE, "bar"), Scalar::full(0.0)},
                 {VariableName(FORCES, "temperature"), Scalar::full(15.0)},
                 {VariableName(FORCES, "t"), Scalar::full(1.3)},
                 {VariableName(OLD_FORCES, "t"), Scalar::full(1.1)}};
  auto out = model->value(in);
  REQUIRE(model->last_iterations() == 7);

  // Re-run the update with the solution being the initial guess
  for (auto && [vname, var] : out)
    in[vname] = var;
  model->value(in);
  REQUIRE(model->last_iterations() == 0);
}
