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
#include "neml2/base/NEML2Object.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

#ifdef NEML2_HAS_CSV

TEST_CASE("MultiColumnCSVScalar", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_MultiColumnCSVScalar.i");
  const auto full_csv = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
  const auto partial_csv = Scalar::create({{1, 4}, {2, 5}});
  const auto full_size = TensorShape{3, 2};
  const auto partial_size = TensorShape{2, 2};
  const auto base_size = TensorShape{};

  SECTION("Parse CSV by column correctly")
  {
    const auto a = factory->get_object<Scalar>("Tensors", "a");
    REQUIRE(at::allclose(*a, full_csv));
    REQUIRE(a->batch_sizes() == full_size);
    REQUIRE(a->base_sizes() == base_size);

    const auto b = factory->get_object<Scalar>("Tensors", "b");
    REQUIRE(at::allclose(*b, full_csv));
    REQUIRE(b->batch_sizes() == full_size);
    REQUIRE(b->base_sizes() == base_size);

    const auto c = factory->get_object<Scalar>("Tensors", "c");
    REQUIRE(at::allclose(*c, partial_csv));
    REQUIRE(c->batch_sizes() == partial_size);
    REQUIRE(c->base_sizes() == base_size);

    const auto d = factory->get_object<Scalar>("Tensors", "d");
    REQUIRE(at::allclose(*d, partial_csv));
    REQUIRE(d->batch_sizes() == partial_size);
    REQUIRE(d->base_sizes() == base_size);

    const auto e = factory->get_object<Scalar>("Tensors", "e");
    REQUIRE(at::allclose(*e, full_csv));
    REQUIRE(e->batch_sizes() == full_size);
    REQUIRE(e->base_sizes() == base_size);

    const auto f = factory->get_object<Scalar>("Tensors", "f");
    REQUIRE(at::allclose(*f, partial_csv));
    REQUIRE(f->batch_sizes() == partial_size);
    REQUIRE(f->base_sizes() == base_size);
  }

  SECTION("Error on invalid user options")
  {
    REQUIRE_THROWS_WITH(factory->get_object<Tensor>("Tensors", "g"),
                        Catch::Matchers::ContainsSubstring(
                            "Only one of column_names or column_indices can be set."));
    REQUIRE_THROWS_WITH(
        factory->get_object<Tensor>("Tensors", "h"),
        Catch::Matchers::ContainsSubstring(
            "If there is no header row, column_names cannot be used. Use column_indices intead."));
    REQUIRE_THROWS_WITH(
        factory->get_object<Tensor>("Tensors", "i"),
        Catch::Matchers::ContainsSubstring("Either column_names or column_indices must be set."));
  }
}

#endif