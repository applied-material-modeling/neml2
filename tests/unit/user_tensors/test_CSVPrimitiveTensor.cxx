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

TEST_CASE("CSVPrimitiveTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_CSVPrimitiveTensor.i");
  const auto tensor_size = TensorShape{4};

  SECTION("read_all")
  {
    SECTION("vector")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "all_columns_vector");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }

    SECTION("SR2")
    {
      const auto parsed = factory->get_object<SR2>("Tensors", "all_columns_SR2");
      const auto expected =
          SR2::create({{2.0, 3.0, 4.0, 5.0 * sqrt(2), 6.0 * sqrt(2), 7.0 * sqrt(2)},
                       {3.0, 4.0, 5.0, 6.0 * sqrt(2), 7.0 * sqrt(2), 8.0 * sqrt(2)},
                       {4.0, 5.0, 6.0, 7.0 * sqrt(2), 8.0 * sqrt(2), 9.0 * sqrt(2)},
                       {5.0, 6.0, 7.0, 8.0 * sqrt(2), 9.0 * sqrt(2), 10.0 * sqrt(2)}});
      REQUIRE(at::allclose(*parsed, expected));
    }
  }

  SECTION("read_by_indices")
  {
    SECTION("scalar")
    {
      const auto parsed = factory->get_object<Scalar>("Tensors", "scalar");
      const auto expected = Scalar::create({0, 1, 2, 3});
      REQUIRE(at::allclose(*parsed, expected));
    }

    SECTION("SR2")
    {
      const auto parsed = factory->get_object<SR2>("Tensors", "SR2");
      const auto expected =
          SR2::create({{2.0, 3.0, 4.0, 5.0 * sqrt(2), 6.0 * sqrt(2), 7.0 * sqrt(2)},
                       {3.0, 4.0, 5.0, 6.0 * sqrt(2), 7.0 * sqrt(2), 8.0 * sqrt(2)},
                       {4.0, 5.0, 6.0, 7.0 * sqrt(2), 8.0 * sqrt(2), 9.0 * sqrt(2)},
                       {5.0, 6.0, 7.0, 8.0 * sqrt(2), 9.0 * sqrt(2), 10.0 * sqrt(2)}});
      REQUIRE(at::allclose(*parsed, expected));
    }
  }

  SECTION("parse_format")
  {
    SECTION("delimiter")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "delimiter");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("starting_row")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "starting_row");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("no_header")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "no_header");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("starting_row + no_header")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "starting_row_no_header");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}});
      REQUIRE(at::allclose(*parsed, expected));
    }
    SECTION("no_header + indices")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "no_header_indices");
      const auto expected = Vec::create({{3, 2, 1}, {4, 3, 2}, {5, 4, 3}, {6, 5, 4}});
      REQUIRE(at::allclose(*parsed, expected));
    }
  }

  SECTION("batch_shape")
  {
    const auto parsed = factory->get_object<Vec>("Tensors", "batch_shape");
    const auto expected = Vec::create({{{1, 2, 3}, {2, 3, 4}}, {{3, 4, 5}, {4, 5, 6}}});
    REQUIRE(at::allclose(*parsed, expected));
  }

  SECTION("errors")
  {
    SECTION("invalid user options")
    {
      REQUIRE_THROWS_WITH(factory->get_object<Scalar>("Tensors", "error_col_name_col_ind"),
                          Catch::Matchers::ContainsSubstring(
                              "Only one of column_names or column_indices can be set."));

      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_name_header"),
          Catch::Matchers::ContainsSubstring("no_header is set to true, column_names cannot be "
                                             "used. Use column_indices instead."));

      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_batch_shape"),
                          Catch::Matchers::ContainsSubstring(
                              "The requested batch_shape [2, 3] is incompatible with the "
                              "number of values read from the CSV file (12)."));

      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_starting_row"),
                          Catch::Matchers::ContainsSubstring("starting_row must be non-negative"));
    }

    SECTION("column not in file")
    {
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_name"),
          Catch::Matchers::ContainsSubstring("Column name disp_w does not exist in CSV file."));

      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_col_ind"),
                          Catch::Matchers::ContainsSubstring("Column index 3 is out of bounds."));
    }

    SECTION("non-numeric data")
    {
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_non_numeric_read_all_header"),
                          Catch::Matchers::ContainsSubstring(
                              "Non-numeric value found in CSV file at row 3, column disp_y"));

      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_non_numeric_read_all_no_header"),
          Catch::Matchers::ContainsSubstring(
              "Non-numeric value found in CSV file at row 3, in column with index 1"));

      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_non_numeric_read_ind_no_header"),
          Catch::Matchers::ContainsSubstring(
              "Non-numeric value found in CSV file at row 3, in column with index 1"));

      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_non_numeric_read_ind_header"),
                          Catch::Matchers::ContainsSubstring(
                              "Non-numeric value found in CSV file at row 3, column s_yy"));
    }

    SECTION("invalid number of columns")
    {
      REQUIRE_THROWS_WITH(
          factory->get_object<Scalar>("Tensors", "error_col_name_component_mismatch"),
          Catch::Matchers::ContainsSubstring("Number of column_names provided (2) does not match "
                                             "the expected number of components in Scalar (1)."));

      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_ind_component_mismatch"),
          Catch::Matchers::ContainsSubstring("Number of column_indices provided (2) does not match "
                                             "the expected number of components in Vec (3)."));

      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_no_component_mismatch"),
          Catch::Matchers::ContainsSubstring("Number of columns provided (2) does not match "
                                             "the expected number of components in Vec (3)."));

      REQUIRE_THROWS_WITH(
          factory->get_object<SR2>("Tensors", "error_col_no_component_mismatch_SR2"),
          Catch::Matchers::ContainsSubstring("Number of columns provided (5) does not match "
                                             "the expected number of components in SR2 (6)."));
    }
  }
}

#endif