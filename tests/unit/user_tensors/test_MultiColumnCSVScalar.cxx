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
#include "neml2/tensors/Scalar.h"
#include "neml2/base/NEML2Object.h"

using namespace neml2;

#ifdef NEML2_HAS_CSV

TEST_CASE("MultiColumnCSVScalar", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_MultiColumnCSVScalar.i");

  SECTION("read_all")
  {
    const auto parsed = factory->get_object<Scalar>("Tensors", "all_columns");
    const auto expected = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
    REQUIRE(at::allclose(*parsed, expected));
  }

  SECTION("read_by_indices")
  {
    SECTION("column_names")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "column_names");
      const auto expected = Scalar::create({{3, 6}, {1, 4}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("column_indices")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "column_indices");
      const auto expected = Scalar::create({{3, 6}, {1, 4}});
      REQUIRE(at::allclose(*parsed, expected));
    }
  }

  SECTION("parse_format")
  {
    SECTION("delimiter")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "delimiter");
      const auto expected = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("starting_row")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "starting_row");
      const auto expected = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("no_header")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "no_header");
      const auto expected = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("starting_row + no_header")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "starting_row_no_header");
      const auto expected = Scalar::create({{1, 4}, {2, 5}, {3, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("no_header + indices")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "no_header_indices");
      const auto expected = Scalar::create({{3, 6}, {1, 4}});
      REQUIRE(at::allclose(*parsed, expected));
    }
  }

  SECTION("indexing")
  {
    const auto parsed = factory->get_object<Scalar>("Tensors", "row_major");
    const auto expected = Scalar::create({{1, 2, 3}, {4, 5, 6}});
    REQUIRE(at::allclose(*parsed, expected));
  }

  SECTION("batch_shape")
  {
    const auto parsed = factory->get_object<Scalar>("Tensors", "batch_shape");
    const auto expected = Scalar::create({{{1}, {4}, {2}}, {{5}, {3}, {6}}});
    REQUIRE(at::allclose(*parsed, expected));
  }

  SECTION("errors")
  {
    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_1"),
                        Catch::Matchers::ContainsSubstring(
                            "Only one of column_names or column_indices can be set."));

    REQUIRE_THROWS_WITH(
        factory->get_object<Scalar>("Tensors", "error_2"),
        Catch::Matchers::ContainsSubstring(
            "no_header is set to true, column_names cannot be used. Use column_indices instead."));

    REQUIRE_THROWS_WITH(
        factory->get_object<Scalar>("Tensors", "error_3"),
        Catch::Matchers::ContainsSubstring("Column name col4 does not exist in CSV file."));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_4"),
                        Catch::Matchers::ContainsSubstring("Column index 3 is out of bounds."));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_5"),
                        Catch::Matchers::ContainsSubstring("Column index 3 is out of bounds."));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_6"),
                        Catch::Matchers::ContainsSubstring("Non-numeric value found in CSV file"));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_7"),
                        Catch::Matchers::ContainsSubstring("Non-numeric value found in CSV file"));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_8"),
                        Catch::Matchers::ContainsSubstring("Non-numeric value found in CSV file"));

    REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_9"),
                        Catch::Matchers::ContainsSubstring(
                            "The requested batch_shape [5, 8] is incompatible with the "
                            "number of values read from the CSV file (6)."));
  }
}

#endif
