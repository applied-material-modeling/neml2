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
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include "neml2/base/LabeledAxis.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;
using SizePair = std::pair<Size, Size>;

struct EqualsSlice : Catch::Matchers::MatcherGenericBase
{
  EqualsSlice(const indexing::Slice & slice)
    : slice{slice}
  {
  }

  bool match(const indexing::Slice & other) const
  {
    return slice.start() == other.start() && slice.stop() == other.stop() &&
           slice.step() == other.step();
  }

  std::string describe() const override { return "Equals: " + utils::stringify(slice); }

private:
  const indexing::Slice & slice;
};

TEST_CASE("LabeledAxis", "[models]")
{
  SECTION("class LabeledAxis")
  {
    LabeledAxis a;
    a.add_variable<Scalar>("scalar");
    a.add_variable<SR2>("r2t");
    a.add_subaxis("sub1");
    a.subaxis("sub1").add_variable<Scalar>("scalar");
    a.subaxis("sub1").add_variable<SR2>("r2t");
    a.subaxis("sub1").add_subaxis("sub2");
    a.subaxis("sub1").subaxis("sub2").add_variable<Scalar>("scalar");
    a.subaxis("sub1").subaxis("sub2").add_variable<SR2>("r2t");
    a.subaxis("sub1").add_subaxis("sub3");
    a.subaxis("sub1").subaxis("sub3").add_variable<Scalar>("scalar");
    a.subaxis("sub1").subaxis("sub3").add_variable("foo", 5);
    const auto & sub1 = a.subaxis("sub1");
    const auto & sub2 = sub1.subaxis("sub2");
    const auto & sub3 = sub1.subaxis("sub3");

    auto setup = GENERATE(true, false);

    if (setup)
      a.setup_layout();

    SECTION("is_setup") { REQUIRE(a.is_setup() == setup); }

    SECTION("size")
    {
      REQUIRE(a.size() == 27);
      REQUIRE(a.size("r2t") == 6);
      REQUIRE(a.size("scalar") == 1);
      REQUIRE(a.size({"sub1", "r2t"}) == 6);
      REQUIRE(a.size({"sub1", "scalar"}) == 1);
      REQUIRE(a.size({"sub1", "sub2", "r2t"}) == 6);
      REQUIRE(a.size({"sub1", "sub2", "scalar"}) == 1);
      REQUIRE(a.size({"sub1", "sub3", "foo"}) == 5);
      REQUIRE(a.size({"sub1", "sub3", "scalar"}) == 1);
      REQUIRE(a.size("sub1") == 20);
      REQUIRE(a.size({"sub1", "sub2"}) == 7);
      REQUIRE(a.size({"sub1", "sub3"}) == 6);

      REQUIRE(sub1.size() == 20);
      REQUIRE(sub1.size("r2t") == 6);
      REQUIRE(sub1.size("scalar") == 1);
      REQUIRE(sub1.size({"sub2", "r2t"}) == 6);
      REQUIRE(sub1.size({"sub2", "scalar"}) == 1);
      REQUIRE(sub1.size({"sub3", "foo"}) == 5);
      REQUIRE(sub1.size({"sub3", "scalar"}) == 1);
      REQUIRE(sub1.size("sub2") == 7);
      REQUIRE(sub1.size("sub3") == 6);

      REQUIRE(sub2.size() == 7);
      REQUIRE(sub2.size("r2t") == 6);
      REQUIRE(sub2.size("scalar") == 1);

      REQUIRE(sub3.size() == 6);
      REQUIRE(sub3.size("foo") == 5);
      REQUIRE(sub3.size("scalar") == 1);
    }

    SECTION("slice")
    {
      if (setup)
      {
        REQUIRE_THAT(a.slice("r2t"), EqualsSlice(indexing::Slice(0, 6)));
        REQUIRE_THAT(a.slice("scalar"), EqualsSlice(indexing::Slice(6, 7)));
        REQUIRE_THAT(a.slice({"sub1", "r2t"}), EqualsSlice(indexing::Slice(7, 13)));
        REQUIRE_THAT(a.slice({"sub1", "scalar"}), EqualsSlice(indexing::Slice(13, 14)));
        REQUIRE_THAT(a.slice({"sub1", "sub2", "r2t"}), EqualsSlice(indexing::Slice(14, 20)));
        REQUIRE_THAT(a.slice({"sub1", "sub2", "scalar"}), EqualsSlice(indexing::Slice(20, 21)));
        REQUIRE_THAT(a.slice({"sub1", "sub3", "foo"}), EqualsSlice(indexing::Slice(21, 26)));
        REQUIRE_THAT(a.slice({"sub1", "sub3", "scalar"}), EqualsSlice(indexing::Slice(26, 27)));
        REQUIRE_THAT(a.slice("sub1"), EqualsSlice(indexing::Slice(7, 27)));
        REQUIRE_THAT(a.slice({"sub1", "sub2"}), EqualsSlice(indexing::Slice(14, 21)));
        REQUIRE_THAT(a.slice({"sub1", "sub3"}), EqualsSlice(indexing::Slice(21, 27)));

        REQUIRE_THAT(sub1.slice("r2t"), EqualsSlice(indexing::Slice(0, 6)));
        REQUIRE_THAT(sub1.slice("scalar"), EqualsSlice(indexing::Slice(6, 7)));
        REQUIRE_THAT(sub1.slice({"sub2", "r2t"}), EqualsSlice(indexing::Slice(7, 13)));
        REQUIRE_THAT(sub1.slice({"sub2", "scalar"}), EqualsSlice(indexing::Slice(13, 14)));
        REQUIRE_THAT(sub1.slice({"sub3", "foo"}), EqualsSlice(indexing::Slice(14, 19)));
        REQUIRE_THAT(sub1.slice({"sub3", "scalar"}), EqualsSlice(indexing::Slice(19, 20)));
        REQUIRE_THAT(sub1.slice("sub2"), EqualsSlice(indexing::Slice(7, 14)));
        REQUIRE_THAT(sub1.slice("sub3"), EqualsSlice(indexing::Slice(14, 20)));

        REQUIRE_THAT(sub2.slice("r2t"), EqualsSlice(indexing::Slice(0, 6)));
        REQUIRE_THAT(sub2.slice("scalar"), EqualsSlice(indexing::Slice(6, 7)));

        REQUIRE_THAT(sub3.slice("foo"), EqualsSlice(indexing::Slice(0, 5)));
        REQUIRE_THAT(sub3.slice("scalar"), EqualsSlice(indexing::Slice(5, 6)));
      }
    }

    SECTION("nvariable")
    {
      REQUIRE(a.nvariable() == 8);
      REQUIRE(sub1.nvariable() == 6);
      REQUIRE(sub2.nvariable() == 2);
      REQUIRE(sub3.nvariable() == 2);
    }

    SECTION("has_variable")
    {
      REQUIRE(a.has_variable("r2t"));
      REQUIRE(a.has_variable("scalar"));
      REQUIRE(!a.has_variable("foo"));
      REQUIRE(sub1.has_variable("r2t"));
      REQUIRE(!sub1.has_variable("foo"));
    }

    SECTION("variable_id")
    {
      if (setup)
      {
        REQUIRE(a.variable_id("r2t") == 0);
        REQUIRE(a.variable_id("scalar") == 1);
        REQUIRE(a.variable_id({"sub1", "r2t"}) == 2);
        REQUIRE(a.variable_id({"sub1", "scalar"}) == 3);
        REQUIRE(a.variable_id({"sub1", "sub2", "r2t"}) == 4);
        REQUIRE(a.variable_id({"sub1", "sub2", "scalar"}) == 5);
        REQUIRE(a.variable_id({"sub1", "sub3", "foo"}) == 6);
        REQUIRE(a.variable_id({"sub1", "sub3", "scalar"}) == 7);

        REQUIRE(sub1.variable_id("r2t") == 0);
        REQUIRE(sub1.variable_id("scalar") == 1);
        REQUIRE(sub1.variable_id({"sub2", "r2t"}) == 2);
        REQUIRE(sub1.variable_id({"sub2", "scalar"}) == 3);
        REQUIRE(sub1.variable_id({"sub3", "foo"}) == 4);
        REQUIRE(sub1.variable_id({"sub3", "scalar"}) == 5);

        REQUIRE(sub2.variable_id("r2t") == 0);
        REQUIRE(sub2.variable_id("scalar") == 1);

        REQUIRE(sub3.variable_id("foo") == 0);
        REQUIRE(sub3.variable_id("scalar") == 1);
      }
    }

    SECTION("variable_names")
    {
      if (setup)
      {
        REQUIRE(a.variable_names() == std::vector<LabeledAxisAccessor>{"r2t",
                                                                       "scalar",
                                                                       {"sub1", "r2t"},
                                                                       {"sub1", "scalar"},
                                                                       {"sub1", "sub2", "r2t"},
                                                                       {"sub1", "sub2", "scalar"},
                                                                       {"sub1", "sub3", "foo"},
                                                                       {"sub1", "sub3", "scalar"}});
        REQUIRE(sub1.variable_names() == std::vector<LabeledAxisAccessor>{"r2t",
                                                                          "scalar",
                                                                          {"sub2", "r2t"},
                                                                          {"sub2", "scalar"},
                                                                          {"sub3", "foo"},
                                                                          {"sub3", "scalar"}});
        REQUIRE(sub2.variable_names() == std::vector<LabeledAxisAccessor>{"r2t", "scalar"});
        REQUIRE(sub3.variable_names() == std::vector<LabeledAxisAccessor>{"foo", "scalar"});
      }
    }

    SECTION("variable_slices")
    {
      if (setup)
      {
        REQUIRE(a.variable_slices()[0] == SizePair{0, 6});
        REQUIRE(a.variable_slices()[1] == SizePair{6, 7});
        REQUIRE(a.variable_slices()[2] == SizePair{7, 13});
        REQUIRE(a.variable_slices()[3] == SizePair{13, 14});
        REQUIRE(a.variable_slices()[4] == SizePair{14, 20});
        REQUIRE(a.variable_slices()[5] == SizePair{20, 21});
        REQUIRE(a.variable_slices()[6] == SizePair{21, 26});
        REQUIRE(a.variable_slices()[7] == SizePair{26, 27});

        REQUIRE(sub1.variable_slices()[0] == SizePair{0, 6});
        REQUIRE(sub1.variable_slices()[1] == SizePair{6, 7});
        REQUIRE(sub1.variable_slices()[2] == SizePair{7, 13});
        REQUIRE(sub1.variable_slices()[3] == SizePair{13, 14});

        REQUIRE(sub2.variable_slices()[0] == SizePair{0, 6});
        REQUIRE(sub2.variable_slices()[1] == SizePair{6, 7});

        REQUIRE(sub3.variable_slices()[0] == SizePair{0, 5});
        REQUIRE(sub3.variable_slices()[1] == SizePair{5, 6});
      }
    }

    SECTION("variable_slice")
    {
      if (setup)
      {
        REQUIRE(a.variable_slice("r2t") == SizePair{0, 6});
        REQUIRE(a.variable_slice("scalar") == SizePair{6, 7});
        REQUIRE(a.variable_slice({"sub1", "r2t"}) == SizePair{7, 13});
        REQUIRE(a.variable_slice({"sub1", "scalar"}) == SizePair{13, 14});
        REQUIRE(a.variable_slice({"sub1", "sub2", "r2t"}) == SizePair{14, 20});
        REQUIRE(a.variable_slice({"sub1", "sub2", "scalar"}) == SizePair{20, 21});
        REQUIRE(a.variable_slice({"sub1", "sub3", "foo"}) == SizePair{21, 26});
        REQUIRE(a.variable_slice({"sub1", "sub3", "scalar"}) == SizePair{26, 27});

        REQUIRE(sub1.variable_slice("r2t") == SizePair{0, 6});
        REQUIRE(sub1.variable_slice("scalar") == SizePair{6, 7});
        REQUIRE(sub1.variable_slice({"sub2", "r2t"}) == SizePair{7, 13});
        REQUIRE(sub1.variable_slice({"sub2", "scalar"}) == SizePair{13, 14});
        REQUIRE(sub1.variable_slice({"sub3", "foo"}) == SizePair{14, 19});
        REQUIRE(sub1.variable_slice({"sub3", "scalar"}) == SizePair{19, 20});

        REQUIRE(sub2.variable_slice("r2t") == SizePair{0, 6});
        REQUIRE(sub2.variable_slice("scalar") == SizePair{6, 7});

        REQUIRE(sub3.variable_slice("foo") == SizePair{0, 5});
        REQUIRE(sub3.variable_slice("scalar") == SizePair{5, 6});
      }
    }

    SECTION("variable_sizes")
    {
      if (setup)
      {
        REQUIRE(a.variable_sizes() == std::vector<Size>{6, 1, 6, 1, 6, 1, 5, 1});
        REQUIRE(sub1.variable_sizes() == std::vector<Size>{6, 1, 6, 1, 5, 1});
        REQUIRE(sub2.variable_sizes() == std::vector<Size>{6, 1});
        REQUIRE(sub3.variable_sizes() == std::vector<Size>{5, 1});
      }
    }

    SECTION("variable_size")
    {
      REQUIRE(a.variable_size("r2t") == 6);
      REQUIRE(a.variable_size("scalar") == 1);
      REQUIRE(a.variable_size({"sub1", "r2t"}) == 6);
      REQUIRE(a.variable_size({"sub1", "scalar"}) == 1);
      REQUIRE(a.variable_size({"sub1", "sub2", "r2t"}) == 6);
      REQUIRE(a.variable_size({"sub1", "sub2", "scalar"}) == 1);
      REQUIRE(a.variable_size({"sub1", "sub3", "foo"}) == 5);
      REQUIRE(a.variable_size({"sub1", "sub3", "scalar"}) == 1);

      REQUIRE(sub1.variable_size("r2t") == 6);
      REQUIRE(sub1.variable_size("scalar") == 1);
      REQUIRE(sub1.variable_size({"sub2", "r2t"}) == 6);
      REQUIRE(sub1.variable_size({"sub2", "scalar"}) == 1);
      REQUIRE(sub1.variable_size({"sub3", "foo"}) == 5);
      REQUIRE(sub1.variable_size({"sub3", "scalar"}) == 1);

      REQUIRE(sub2.variable_size("r2t") == 6);
      REQUIRE(sub2.variable_size("scalar") == 1);

      REQUIRE(sub3.variable_size("foo") == 5);
      REQUIRE(sub3.variable_size("scalar") == 1);
    }

    SECTION("nsubaxis")
    {
      REQUIRE(a.nsubaxis() == 1);
      REQUIRE(sub1.nsubaxis() == 2);
      REQUIRE(sub2.nsubaxis() == 0);
      REQUIRE(sub3.nsubaxis() == 0);
    }

    SECTION("has_subaxis")
    {
      REQUIRE(a.has_subaxis("sub1"));
      REQUIRE(a.has_subaxis({"sub1", "sub2"}));
      REQUIRE(a.has_subaxis({"sub1", "sub3"}));
      REQUIRE(!a.has_subaxis("sub2"));
      REQUIRE(sub1.has_subaxis("sub2"));
      REQUIRE(sub1.has_subaxis("sub3"));
      REQUIRE(!sub1.has_subaxis("sub4"));
    }

    SECTION("subaxis_id")
    {
      if (setup)
      {
        REQUIRE(a.subaxis_id("sub1") == 0);
        REQUIRE(sub1.subaxis_id("sub2") == 0);
        REQUIRE(sub1.subaxis_id("sub3") == 1);
      }
    }

    SECTION("subaxes")
    {
      if (setup)
      {
        REQUIRE(a.subaxes() == std::vector<const LabeledAxis *>{&sub1});
        REQUIRE(sub1.subaxes() == std::vector<const LabeledAxis *>{&sub2, &sub3});
      }
    }

    SECTION("subaxis")
    {
      REQUIRE(&a.subaxis("sub1") == &sub1);
      REQUIRE(&a.subaxis({"sub1", "sub2"}) == &sub2);
      REQUIRE(&a.subaxis({"sub1", "sub3"}) == &sub3);
      REQUIRE(&sub1.subaxis("sub2") == &sub2);
      REQUIRE(&sub1.subaxis("sub3") == &sub3);
    }

    SECTION("subaxis_names")
    {
      if (setup)
      {
        REQUIRE(a.subaxis_names() == std::vector<std::string>{"sub1"});
        REQUIRE(sub1.subaxis_names() == std::vector<std::string>{"sub2", "sub3"});
      }
    }

    SECTION("subaxis_slices")
    {
      if (setup)
      {
        REQUIRE(a.subaxis_slices()[0] == SizePair{7, 27});
        REQUIRE(sub1.subaxis_slices()[0] == SizePair{7, 14});
        REQUIRE(sub1.subaxis_slices()[1] == SizePair{14, 20});
      }
    }

    SECTION("subaxis_slice")
    {
      if (setup)
      {
        REQUIRE(a.subaxis_slice("sub1") == SizePair{7, 27});
        REQUIRE(a.subaxis_slice({"sub1", "sub2"}) == SizePair{14, 21});
        REQUIRE(a.subaxis_slice({"sub1", "sub3"}) == SizePair{21, 27});
        REQUIRE(sub1.subaxis_slice("sub2") == SizePair{7, 14});
        REQUIRE(sub1.subaxis_slice("sub3") == SizePair{14, 20});
      }
    }

    SECTION("subaxis_sizes")
    {
      if (setup)
      {
        REQUIRE(a.subaxis_sizes() == std::vector<Size>{20});
        REQUIRE(sub1.subaxis_sizes() == std::vector<Size>{7, 6});
      }
    }

    SECTION("subaxis_size")
    {
      REQUIRE(a.subaxis_size("sub1") == 20);
      REQUIRE(a.subaxis_size({"sub1", "sub2"}) == 7);
      REQUIRE(a.subaxis_size({"sub1", "sub3"}) == 6);
      REQUIRE(sub1.subaxis_size("sub2") == 7);
      REQUIRE(sub1.subaxis_size("sub3") == 6);
    }
  }

  SECTION("operator==")
  {
    // test1 and test2 are equal:
    // They have the same set of items with same names and storage sizes.
    LabeledAxis test1;
    test1.add_variable<Scalar>("scalar1");
    test1.add_variable<SR2>("r2t1");
    test1.add_variable<Scalar>("scalar2");
    test1.add_variable<SR2>("r2t2");
    test1.add_variable<Scalar>("scalar3");
    test1.setup_layout();

    LabeledAxis test2;
    test2.add_variable<SR2>("r2t1");
    test2.add_variable<Scalar>("scalar3");
    test2.add_variable<SR2>("r2t2");
    test2.add_variable<Scalar>("scalar2");
    test2.add_variable<Scalar>("scalar1");
    test2.setup_layout();

    REQUIRE(test1 == test2);
  }

  SECTION("operator!=")
  {
    // test1 and test2 are equal:
    // They have the same set of items with same names and storage sizes.
    LabeledAxis test1;
    test1.add_variable<Scalar>("scalar1");
    test1.add_variable<SR2>("r2t1");
    test1.add_variable<Scalar>("scalar2");
    test1.add_variable<SR2>("r2t2");
    test1.add_variable<Scalar>("scalar3");
    test1.setup_layout();

    LabeledAxis test2;
    test2.add_variable<SR2>("r2t1");
    test2.add_variable<Scalar>("scalar3");
    test2.add_variable<SR2>("r2t2");
    test2.add_variable<Scalar>("scalar2");
    test2.add_variable<Scalar>("scalar1");
    test2.setup_layout();

    // test3 is NOT equal to test1 nor test2
    LabeledAxis test3;
    test3.add_variable<SR2>("r2t1");
    test3.add_variable<Scalar>("scalar3");
    test3.add_variable<SR2>("r2t2");
    test3.setup_layout();

    REQUIRE(test1 != test3);
    REQUIRE(test2 != test3);
  }
}
