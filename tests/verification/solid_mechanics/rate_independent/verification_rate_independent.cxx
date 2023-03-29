// Copyright 2023, UChicago Argonne, LLC
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

#include <catch2/catch.hpp>

#include "StructuralDriver.h"
#include "neml2/misc/math.h"
#include "TestUtils.h"

#include "VerificationTest.h"

using namespace neml2;

TEST_CASE("Rate independent perfect plasticity verification",
          "[StructuralVerificationTests][RIPerfect]")
{
  load_model("verification/solid_mechanics/rate_independent/perfect.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Rate independent perfect plasticity, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/rate_independent/perfect.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}

TEST_CASE("Rate independent Voce isotropic hardening verification",
          "[StructuralVerificationTests][RIVoce]")
{
  load_model("verification/solid_mechanics/rate_independent/voceiso.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Rate independent Voce hardening, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/rate_independent/voceiso.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}

TEST_CASE("Rate independent isotropic + kinematic hardening verification",
          "[StructuralVerificationTests][RIVoceLinKin]")
{
  load_model("verification/solid_mechanics/rate_independent/voceisolinkin.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Rate independent Voce + linear kinematic hardening, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/rate_independent/voceisolinkin.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}
