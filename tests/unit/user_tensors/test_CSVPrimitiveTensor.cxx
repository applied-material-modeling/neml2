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
#include "neml2/base/NEML2Object.h"
#include "neml2/tensors/tensors.h"

#include "utils.h"

using namespace neml2;

#ifdef NEML2_CSV

TEST_CASE("CSVPrimitiveTensor", "[user_tensors]")
{
  auto factory = load_input("user_tensors/test_CSVPrimitiveTensor.i");
  const auto tensor_size = TensorShape{4};

  SECTION("read_all")
  {
    const auto parsed = factory->get_object<Vec>("Tensors", "all_columns_vector");
    const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}});
    REQUIRE_THAT(*parsed, test::allclose(expected));
  }

  SECTION("read_by_indices")
  {
    const auto parsed = factory->get_object<Scalar>("Tensors", "scalar");
    const auto expected = Scalar::create({0, 1});
    REQUIRE_THAT(*parsed, test::allclose(expected));
  }

  SECTION("multiply_factor")
  {
    const auto parsed = factory->get_object<SR2>("Tensors", "all_columns_SR2");
    const auto expected =
        SR2::create({{2.0, 3.0, 4.0, 5.0 * sqrt(2), 6.0 * sqrt(2), 7.0 * sqrt(2)},
                     {3.0, 4.0, 5.0, 6.0 * sqrt(2), 7.0 * sqrt(2), 8.0 * sqrt(2)}});
    REQUIRE_THAT(*parsed, test::allclose(expected));
  }

  SECTION("parse_format")
  {
    SECTION("delimiter")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "delimiter");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}});
      REQUIRE_THAT(*parsed, test::allclose(expected));
    }
    SECTION("starting_row")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "starting_row");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}});
      REQUIRE_THAT(*parsed, test::allclose(expected));
    }
    SECTION("no_header")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "no_header");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}});
      REQUIRE_THAT(*parsed, test::allclose(expected));
    }
    SECTION("starting_row + no_header")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "starting_row_no_header");
      const auto expected = Vec::create({{1, 2, 3}, {2, 3, 4}});
      REQUIRE_THAT(*parsed, test::allclose(expected));
    }
    SECTION("no_header + indices")
    {
      const auto parsed = factory->get_object<Vec>("Tensors", "no_header_indices");
      const auto expected = Vec::create({{3, 2, 1}, {4, 3, 2}});
      REQUIRE_THAT(*parsed, test::allclose(expected));
    }
  }

  SECTION("errors")
  {
    SECTION("parse_format_error")
    {
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_starting_row"),
                          Catch::Matchers::ContainsSubstring("starting_row must be non-negative"));
    }

    SECTION("check_col_errors")
    {
      REQUIRE_THROWS_WITH(factory->get_object<SR2>("Tensors", "error_col_name_component_mismatch"),
                          Catch::Matchers::ContainsSubstring(
                              "Number of columns specified in column_names (2) does not match "
                              "the expected number of components in SR2 (6)."));
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_col_ind_component_mismatch"),
                          Catch::Matchers::ContainsSubstring(
                              "Number of columns specified in column_indices (2) does not match "
                              "the expected number of components in Vec (3)."));
    }

    SECTION("parse_indices_errors")
    {
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_col_name_col_ind"),
                          Catch::Matchers::ContainsSubstring(
                              "Only one of column_names or column_indices can be set."));
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_name_header"),
          Catch::Matchers::ContainsSubstring("no_header is set to true, column_names cannot be "
                                             "used. Use column_indices instead."));
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_name"),
          Catch::Matchers::ContainsSubstring("Column name disp_w does not exist in CSV file."));
    }

    SECTION("read_all_errors")
    {
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_non_numeric_read_all_no_header"),
          Catch::Matchers::ContainsSubstring(
              "Non-numeric value found in CSV file at row 1, in column with index 1"));
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_non_numeric_read_all_header"),
                          Catch::Matchers::ContainsSubstring(
                              "Non-numeric value found in CSV file at row 1, column disp_y"));
    }

    SECTION("read_by_indices_errors")
    {
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_ind_out_of_bounds"),
                          Catch::Matchers::ContainsSubstring("Column index 3 is out of bounds."));
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_non_numeric_read_ind_no_header"),
          Catch::Matchers::ContainsSubstring(
              "Non-numeric value found in CSV file at row 1, in column with index 1"));
      REQUIRE_THROWS_WITH(factory->get_object<Vec>("Tensors", "error_non_numeric_read_ind_header"),
                          Catch::Matchers::ContainsSubstring(
                              "Non-numeric value found in CSV file at row 1, column disp_y"));
    }

    SECTION("invalid number of columns for read all")
    {
      REQUIRE_THROWS_WITH(
          factory->get_object<Vec>("Tensors", "error_col_no_component_mismatch"),
          Catch::Matchers::ContainsSubstring("Number of columns (2) does not match "
                                             "the expected number of components in Vec (3)."));
    }
  }
}

#endif
