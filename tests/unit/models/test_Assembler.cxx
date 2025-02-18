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

#include "neml2/base/Factory.h"
#include "neml2/models/Model.h"
#include "neml2/models/Assembler.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/functions/cat.h"

using namespace neml2;

TEST_CASE("VectorAssembler", "[models]")
{
  auto & model = reload_model("models/SampleRateModel.i", "model");

  const auto & axis = model.input_axis();
  const auto assembler = VectorAssembler(axis);

  const auto T_name = VariableName(FORCES, "temperature");
  const auto foo_name = VariableName(STATE, "foo");
  const auto bar_name = VariableName(STATE, "bar");
  const auto baz_name = VariableName(STATE, "baz");

  const auto T = Scalar::full({2, 5}, 120);
  const auto foo = Scalar::zeros();
  const auto bar = Scalar::full({2, 1}, -1.0);
  const auto baz = SR2::full({5, 2, 5}, 0.1);

  SECTION("assemble_by_variable")
  {
    const auto v = assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
    REQUIRE(v.batch_sizes().concrete() == TensorShape{5, 2, 5});
    REQUIRE(v.base_sizes() == TensorShapeRef{9});
    REQUIRE(at::allclose(v.base_index({axis.slice(T_name)}), T.base_flatten()));
    REQUIRE(at::allclose(v.base_index({axis.slice(foo_name)}), foo.base_flatten()));
    REQUIRE(at::allclose(v.base_index({axis.slice(bar_name)}), bar.base_flatten()));
    REQUIRE(at::allclose(v.base_index({axis.slice(baz_name)}), baz.base_flatten()));
  }

  SECTION("split_by_variable")
  {
    const auto v = assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
    const auto vars = assembler.split_by_variable(v);
    REQUIRE(vars.size() == 4);
    REQUIRE(at::allclose(vars.at(T_name), T.base_flatten()));
    REQUIRE(at::allclose(vars.at(foo_name), foo.base_flatten()));
    REQUIRE(at::allclose(vars.at(bar_name), bar.base_flatten()));
    REQUIRE(at::allclose(vars.at(baz_name), baz.base_flatten()));
  }

  SECTION("split_by_subaxis")
  {
    const auto v = assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
    const auto vs = assembler.split_by_subaxis(v);
    REQUIRE(vs.size() == 2);
    REQUIRE(at::allclose(vs.at(FORCES), T.base_flatten()));
    REQUIRE(at::allclose(vs.at(STATE),
                         base_cat({bar.base_flatten().batch_expand({5, 2, 5}),
                                   baz.base_flatten().batch_expand({5, 2, 5}),
                                   foo.base_flatten().batch_expand({5, 2, 5})})));
  }
}
