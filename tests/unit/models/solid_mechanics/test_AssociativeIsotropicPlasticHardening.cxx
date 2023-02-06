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

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/AssociativeIsotropicPlasticHardening.h"

using namespace neml2;

TEST_CASE("AssociativeIsotropicPlasticHardening", "[AssociativeIsotropicPlasticHardening]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        J2StressMeasure::expected_params() +
                            ParameterSet(KS{"name", "j2"}, KS{"type", "J2StressMeasure"}));
  factory.create_object("Models",
                        IsotropicHardeningYieldFunction::expected_params() +
                            ParameterSet(KS{"name", "yield"},
                                         KS{"type", "IsotropicHardeningYieldFunction"},
                                         KS{"stress_measure", "j2"},
                                         KR{"yield_stress", 100}));
  factory.create_object("Models",
                        AssociativeIsotropicPlasticHardening::expected_params() +
                            ParameterSet(KS{"name", "eprate"},
                                         KS{"type", "AssociativeIsotropicPlasticHardening"},
                                         KS{"yield_function", "yield"}));

  auto & eprate = Factory::get_object<AssociativeIsotropicPlasticHardening>("Models", "eprate");

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(eprate.input().has_subaxis("state"));
    REQUIRE(eprate.input().subaxis("state").has_variable<Scalar>("hardening_rate"));
    REQUIRE(eprate.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(eprate.input().subaxis("state").has_subaxis("hardening_interface"));
    REQUIRE(eprate.input()
                .subaxis("state")
                .subaxis("hardening_interface")
                .has_variable<Scalar>("isotropic_hardening"));

    REQUIRE(eprate.output().has_subaxis("state"));
    REQUIRE(eprate.output()
                .subaxis("state")
                .subaxis("internal_state")
                .has_variable<Scalar>("equivalent_plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, eprate.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").set(Scalar(0.01, nbatch), "hardening_rate");
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = eprate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, eprate.output(), eprate.input());
    finite_differencing_derivative(
        [eprate](const LabeledVector & x) { return eprate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
