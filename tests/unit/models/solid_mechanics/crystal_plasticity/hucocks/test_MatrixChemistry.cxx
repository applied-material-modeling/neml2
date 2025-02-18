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

#include <filesystem>

#include "utils.h"
#include "neml2/base/Factory.h"
#include "neml2/models/solid_mechanics/crystal_plasticity/hucocks/MatrixChemistry.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;
namespace fs = std::filesystem;

TEST_CASE("MatrixChemistry", "[models/solid_mechanics/crystal_plasticity/hucocks]")
{
  at::manual_seed(42);
  const auto & DTO = default_tensor_options();

  // Load all the models in
  reload_input(
      fs::absolute("models/solid_mechanics/crystal_plasticity/hucocks/test_MatrixChemistry.i"));

  SECTION("Arbitrary example")
  {
    auto & model = Factory::get_object<MatrixChemistry>("Data", "example");

    SECTION("Full list in order")
    {
      auto initial_concentrations = model.initial_concentrations({"Cr", "Ni", "Fe"});
      auto equilibrium_concentrations = model.equilibrium_concentrations({"Cr", "Ni", "Fe"});

      REQUIRE(at::allclose(initial_concentrations, Scalar::create({0.5, 0.3, 0.2}, DTO)));
      REQUIRE(at::allclose(equilibrium_concentrations, Scalar::create({0.4, 0.5, 0.7}, DTO)));
    }
    SECTION("Full list out of order")
    {
      auto initial_concentrations = model.initial_concentrations({"Fe", "Ni", "Cr"});
      auto equilibrium_concentrations = model.equilibrium_concentrations({"Fe", "Ni", "Cr"});

      REQUIRE(at::allclose(initial_concentrations, Scalar::create({0.2, 0.3, 0.5}, DTO)));
      REQUIRE(at::allclose(equilibrium_concentrations, Scalar::create({0.7, 0.5, 0.4}, DTO)));
    }
    SECTION("Subset")
    {
      auto initial_concentrations = model.initial_concentrations({"Fe", "Cr"});
      auto equilibrium_concentrations = model.equilibrium_concentrations({"Fe", "Cr"});

      REQUIRE(at::allclose(initial_concentrations, Scalar::create({0.2, 0.5}, DTO)));
      REQUIRE(at::allclose(equilibrium_concentrations, Scalar::create({0.7, 0.4}, DTO)));
    }
    SECTION("Invalid species")
    {
      REQUIRE_THROWS_WITH(model.initial_concentrations({"Fe", "Cr", "Cu"}),
                          Catch::Matchers::ContainsSubstring("not found in concentrations"));
      REQUIRE_THROWS_WITH(model.equilibrium_concentrations({"Fe", "Cr", "Cu"}),
                          Catch::Matchers::ContainsSubstring("not found in concentrations"));
    }
  }
}
