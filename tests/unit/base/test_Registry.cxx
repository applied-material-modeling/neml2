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

#include "neml2/base/Registry.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/Scalar.h"

TEST_CASE("Registry", "[base]")
{
  SECTION("load")
  {
    const auto & reg = neml2::Registry::get();
    REQUIRE(reg.info().count("FooModel") == 0);

    // Find the dynamic library
    namespace fs = std::filesystem;
    auto pwd = fs::current_path();
    auto lib_dir = pwd / ".." / "extension";
    REQUIRE(fs::exists(lib_dir));
    REQUIRE(fs::is_directory(lib_dir));
    auto lib_so = lib_dir / "libextension.so";
    auto lib_dylib = lib_dir / "libextension.dylib";
    REQUIRE(fs::exists(lib_so) != fs::exists(lib_dylib));

    // Load the library
    if (fs::exists(lib_so))
      neml2::Registry::load(lib_so);
    else
      neml2::Registry::load(lib_dylib);
    REQUIRE(reg.info().count("FooModel") == 1);
  }

  SECTION("load from input")
  {
    namespace fs = std::filesystem;
    auto pwd = fs::current_path();
    auto lib_dir = pwd / ".." / "extension";
    auto & model = neml2::load_model(lib_dir / "FooModel.i", "foo");

    const auto x = neml2::Scalar::full(5);
    const auto out = model.value({{neml2::VariableName("forces", "x"), x}});
    const auto & y = out.at(neml2::VariableName("state", "y"));
    REQUIRE(at::allclose(y, neml2::Scalar::full(5.6)));
  }
}
