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
#include "neml2/tensors/functions/to_assembly.h"
#include "neml2/tensors/functions/from_assembly.h"

#include "unit/tensors/generators.h"
#include "utils.h"

using namespace neml2;

TEST_CASE("Assembler", "[models]")
{
  SECTION("to_assembly<1> from_assembly<1>")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape = test::generate_tensor_shape();

    DYNAMIC_SECTION(shape.desc())
    {
      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = to_assembly<1>({a}, {shape.intmd_sizes}, {shape.base_sizes});

      REQUIRE(b.dynamic_sizes() == a.dynamic_sizes());
      REQUIRE(b.intmd_sizes() == TensorShapeRef{});
      REQUIRE(b.base_sizes() ==
              TensorShapeRef{utils::numel(shape.intmd_sizes) * utils::numel(shape.base_sizes)});

      // make a round trip
      auto c = from_assembly<1>(b, {shape.intmd_sizes}, {shape.base_sizes});
      REQUIRE(a.dynamic_sizes() == c.dynamic_sizes());
      REQUIRE(a.intmd_sizes() == c.intmd_sizes());
      REQUIRE(a.base_sizes() == c.base_sizes());
      REQUIRE_THAT(a, test::allclose(c));
    }
  }

  SECTION("to_assembly<2> from_assembly<2>")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape1 = test::generate_tensor_shape();
    auto shape2 = test::generate_tensor_shape();

    DYNAMIC_SECTION(shape1.desc() << " " << shape2.desc())
    {
      auto shape =
          test::GeneratedTensorShape({2, 3},
                                     utils::add_shapes(shape1.intmd_sizes, shape2.intmd_sizes),
                                     utils::add_shapes(shape1.base_sizes, shape2.base_sizes));

      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = to_assembly<2>(
          {a}, {shape1.intmd_sizes, shape2.intmd_sizes}, {shape1.base_sizes, shape2.base_sizes});

      REQUIRE(b.dynamic_sizes() == a.dynamic_sizes());
      REQUIRE(b.intmd_sizes() == TensorShapeRef{});
      REQUIRE(b.base_sizes() ==
              TensorShapeRef{utils::numel(shape1.intmd_sizes) * utils::numel(shape1.base_sizes),
                             utils::numel(shape2.intmd_sizes) * utils::numel(shape2.base_sizes)});

      // make a round trip
      auto c = from_assembly<2>(
          b, {shape1.intmd_sizes, shape2.intmd_sizes}, {shape1.base_sizes, shape2.base_sizes});
      REQUIRE(a.dynamic_sizes() == c.dynamic_sizes());
      REQUIRE(a.intmd_sizes() == c.intmd_sizes());
      REQUIRE(a.base_sizes() == c.base_sizes());
      REQUIRE_THAT(a, test::allclose(c));
    }
  }

  SECTION("to_assembly<3> from_assembly<3>")
  {
    auto cfg = test::GeneratedTensorConfig(kFloat64, kCPU);
    auto shape1 = test::generate_tensor_shape();
    auto shape2 = test::generate_tensor_shape();
    auto shape3 = test::generate_tensor_shape();

    DYNAMIC_SECTION(shape1.desc() << " " << shape2.desc() << " " << shape3.desc())
    {
      auto shape = test::GeneratedTensorShape(
          {2, 3},
          utils::add_shapes(shape1.intmd_sizes, shape2.intmd_sizes, shape3.intmd_sizes),
          utils::add_shapes(shape1.base_sizes, shape2.base_sizes, shape3.base_sizes));

      auto a = test::generate_random_tensor<Tensor>(cfg, shape);
      auto b = to_assembly<3>({a},
                              {shape1.intmd_sizes, shape2.intmd_sizes, shape3.intmd_sizes},
                              {shape1.base_sizes, shape2.base_sizes, shape3.base_sizes});

      REQUIRE(b.dynamic_sizes() == a.dynamic_sizes());
      REQUIRE(b.intmd_sizes() == TensorShapeRef{});
      REQUIRE(b.base_sizes() ==
              TensorShapeRef{utils::numel(shape1.intmd_sizes) * utils::numel(shape1.base_sizes),
                             utils::numel(shape2.intmd_sizes) * utils::numel(shape2.base_sizes),
                             utils::numel(shape3.intmd_sizes) * utils::numel(shape3.base_sizes)});

      // make a round trip
      auto c = from_assembly<3>(b,
                                {shape1.intmd_sizes, shape2.intmd_sizes, shape3.intmd_sizes},
                                {shape1.base_sizes, shape2.base_sizes, shape3.base_sizes});
      REQUIRE(a.dynamic_sizes() == c.dynamic_sizes());
      REQUIRE(a.intmd_sizes() == c.intmd_sizes());
      REQUIRE(a.base_sizes() == c.base_sizes());
      REQUIRE_THAT(a, test::allclose(c));
    }
  }

  SECTION("VectorAssembler")
  {
    auto model = load_model("models/SampleRateModel.i", "model");

    const auto & axis = model->input_axis();
    const auto assembler = VectorAssembler(axis);

    const auto T_name = VariableName(FORCES, "temperature");
    const auto foo_name = VariableName(STATE, "foo");
    const auto bar_name = VariableName(STATE, "bar");
    const auto baz_name = VariableName(STATE, "baz");

    const auto T = Scalar::full({2, 5}, {}, 120).base_flatten();
    const auto foo = Scalar::zeros().base_flatten();
    const auto bar = Scalar::full({2, 1}, {}, -1.0).base_flatten();
    const auto baz = SR2::full({5, 2, 5}, {}, 0.1).base_flatten();

    SECTION("assemble_by_variable")
    {
      const auto v =
          assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
      REQUIRE(v.batch_sizes().concrete() == TensorShape{5, 2, 5});
      REQUIRE(v.base_sizes() == TensorShapeRef{9});
      REQUIRE(at::allclose(v.base_index({axis.slice(T_name)}), T));
      REQUIRE(at::allclose(v.base_index({axis.slice(foo_name)}), foo));
      REQUIRE(at::allclose(v.base_index({axis.slice(bar_name)}), bar));
      REQUIRE(at::allclose(v.base_index({axis.slice(baz_name)}), baz));
    }

    SECTION("split_by_variable")
    {
      const auto v =
          assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
      const auto vars = assembler.split_by_variable(v);
      REQUIRE(vars.size() == 4);
      REQUIRE(at::allclose(vars.at(T_name), T));
      REQUIRE(at::allclose(vars.at(foo_name), foo));
      REQUIRE(at::allclose(vars.at(bar_name), bar));
      REQUIRE(at::allclose(vars.at(baz_name), baz));
    }

    SECTION("split_by_subaxis")
    {
      const auto v =
          assembler.assemble_by_variable({{T_name, T}, {bar_name, bar}, {baz_name, baz}});
      const auto vs = assembler.split_by_subaxis(v);
      REQUIRE(vs.size() == 2);
      REQUIRE(at::allclose(vs.at(FORCES), T));
      REQUIRE(at::allclose(vs.at(STATE),
                           base_cat({bar.dynamic_expand({5, 2, 5}),
                                     baz.dynamic_expand({5, 2, 5}),
                                     foo.dynamic_expand({5, 2, 5})})));
    }
  }
}
