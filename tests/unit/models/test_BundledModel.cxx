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
#include <filesystem>

#include "neml2/models/BundledModel.h"

using namespace neml2;

TEST_CASE("BundledModel", "[models]")
{
#if !defined(NEML2_HAS_ZLIB)
  SKIP("NEML2 was not compiled with zlib support, cannot package models.");
#elif !defined(NEML2_HAS_JSON)
  SKIP("NEML2 was not compiled with json support, cannot package models.");
#else
  std::ifstream f("models/ComposedModel5.json");
  auto config = nlohmann::json::parse(f);
  bundle_model("models/ComposedModel5.i", "model", "", config, "models/ComposedModel5_model.gz");
  // assert file exists
  std::filesystem::path output_path = "models/ComposedModel5_model.gz";
  REQUIRE(std::filesystem::exists(output_path));
#endif
}
