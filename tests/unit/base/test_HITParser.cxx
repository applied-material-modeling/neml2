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

#include "neml2/base/HITParser.h"
#include "neml2/base/TensorName.h"
#include "neml2/base/EnumSelection.h"

#include "SampleParserTestingModel.h"

using namespace neml2;

TEST_CASE("HITParser", "[base]")
{
  SECTION("parse")
  {
    SECTION("TensorShape")
    {
      REQUIRE(utils::parse<TensorShape>("(1,2,3,4,5,6)") == TensorShape{1, 2, 3, 4, 5, 6});
      REQUIRE(utils::parse<TensorShape>("(1,2,3)") == TensorShape{1, 2, 3});
      REQUIRE(utils::parse<TensorShape>("(1,2,3,)") == TensorShape{1, 2, 3});
      REQUIRE(utils::parse<TensorShape>("(,1,2,3)") == TensorShape{1, 2, 3});
      REQUIRE(utils::parse<TensorShape>("(,1,2,3,)") == TensorShape{1, 2, 3});
      REQUIRE(utils::parse<TensorShape>("( ,  1, 2, 3 , )") == TensorShape{1, 2, 3});
      REQUIRE(utils::parse<TensorShape>("()") == TensorShape{});
      REQUIRE_THROWS_WITH(
          utils::parse<TensorShape>("1"),
          Catch::Matchers::ContainsSubstring("a shape must start with '(' and end with ')'"));
    }

    SECTION("bool")
    {
      REQUIRE(utils::parse<bool>("true"));
      REQUIRE(!utils::parse<bool>("false"));
      REQUIRE_THROWS_WITH(utils::parse<bool>("off"),
                          Catch::Matchers::ContainsSubstring("Failed to parse boolean value"));
    }
  }

  SECTION("class HITParser")
  {
    HITParser parser;
    SECTION("parse")
    {
      auto all_options = parser.parse("base/test_HITParser1.i");
      OptionSet options = all_options["Models"]["foo"];

      SECTION("metadata")
      {
        REQUIRE(options.name() == "foo");
        REQUIRE(options.type() == "SampleParserTestingModel");
        REQUIRE(options.path() == "Models");
        REQUIRE(options.doc() == "This model tests the correctness of parsed options.");
      }

      SECTION("global settings")
      {
        auto & settings = all_options.settings();
        REQUIRE(settings.get<EnumSelection>("default_floating_point_type").as<Dtype>() == kFloat16);
        REQUIRE(settings.get<EnumSelection>("default_integer_type").as<Dtype>() == kInt32);
        REQUIRE(settings.get<std::string>("default_device") == "cuda:1");
        REQUIRE(settings.get<Real>("machine_precision") == Catch::Approx(0.5));
        REQUIRE(settings.get<Real>("tolerance") == Catch::Approx(0.1));
        REQUIRE(settings.get<Real>("tighter_tolerance") == Catch::Approx(0.01));
        REQUIRE(settings.get<std::string>("buffer_name_separator") == "::");
        REQUIRE(settings.get<std::string>("parameter_name_separator") == "::");
      }

      SECTION("booleans")
      {
        REQUIRE(options.get<bool>("bool") == true);
        REQUIRE(options.get<std::vector<bool>>("bool_vec") ==
                std::vector<bool>{true, false, false});
        REQUIRE(options.get<std::vector<std::vector<bool>>>("bool_vec_vec") ==
                std::vector<std::vector<bool>>{
                    {true, false}, {false, true, true}, {false, false, false}});
      }

      SECTION("integers")
      {
        REQUIRE(options.get<int>("int") == 5);
        REQUIRE(options.get<std::vector<int>>("int_vec") == std::vector<int>{5, 6, 7});
        REQUIRE(options.get<std::vector<std::vector<int>>>("int_vec_vec") ==
                std::vector<std::vector<int>>{{-1, 3, -2}, {-5}});
      }

      SECTION("unsigned integers")
      {
        REQUIRE(options.get<unsigned int>("uint") == 30);
        REQUIRE(options.get<std::vector<unsigned int>>("uint_vec") ==
                std::vector<unsigned int>{1, 2, 3});
        REQUIRE(options.get<std::vector<std::vector<unsigned int>>>("uint_vec_vec") ==
                std::vector<std::vector<unsigned int>>{{555}, {123}, {1, 5, 9}});
      }

      SECTION("Reals")
      {
        REQUIRE(options.get<Real>("Real") == Catch::Approx(3.14159));
        REQUIRE_THAT(options.get<std::vector<Real>>("Real_vec"),
                     Catch::Matchers::Approx(std::vector<Real>{-111, 12, 1.1}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[0],
                     Catch::Matchers::Approx(std::vector<Real>{1, 3, 5}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[1],
                     Catch::Matchers::Approx(std::vector<Real>{2, 4, 6}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[2],
                     Catch::Matchers::Approx(std::vector<Real>{-3, -5, -7}));
      }

      SECTION("strings")
      {
        REQUIRE(options.get<std::string>("string") == "today");
        REQUIRE(options.get<std::vector<std::string>>("string_vec") ==
                std::vector<std::string>{"is", "a", "good", "day"});
        REQUIRE(options.get<std::vector<std::vector<std::string>>>("string_vec_vec") ==
                std::vector<std::vector<std::string>>{{"neml2", "is", "very"}, {"useful"}});
      }

      SECTION("TensorShapes")
      {
        auto shape = options.get<TensorShape>("shape");
        auto shape_vec = options.get<std::vector<TensorShape>>("shape_vec");
        auto shape_vec_vec = options.get<std::vector<std::vector<TensorShape>>>("shape_vec_vec");
        REQUIRE(shape == TensorShape{1, 2, 3, 5});
        REQUIRE(shape_vec[0] == TensorShape{1, 2, 3});
        REQUIRE(shape_vec[1] == TensorShape{2, 3});
        REQUIRE(shape_vec[2] == TensorShape{5});
        REQUIRE(shape_vec_vec[0][0] == TensorShape{2, 5});
        REQUIRE(shape_vec_vec[0][1] == TensorShape{});
        REQUIRE(shape_vec_vec[0][2] == TensorShape{3, 3});
        REQUIRE(shape_vec_vec[1][0] == TensorShape{2, 2});
        REQUIRE(shape_vec_vec[1][1] == TensorShape{1});
        REQUIRE(shape_vec_vec[1][2] == TensorShape{22});
      }
    }

    SECTION("error")
    {
      SECTION("setting a suppressed option")
      {
        REQUIRE_THROWS_WITH(parser.parse("base/test_HITParser2.i"),
                            Catch::Matchers::ContainsSubstring(
                                "Option named 'suppressed_option' is suppressed, and its "
                                "value cannot be modified."));
      }
    }
  }
}
