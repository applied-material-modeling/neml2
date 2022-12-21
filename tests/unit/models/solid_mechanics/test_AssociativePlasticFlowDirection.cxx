#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/models/solid_mechanics/J2StressMeasure.h"
#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"

using namespace neml2;

TEST_CASE("AssociativePlasticFlowDirection", "[AssociativePlasticFlowDirection]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 10.0;
  auto sm = std::make_shared<J2StressMeasure>("stress_measure");
  auto yield = std::make_shared<YieldFunction>("yield_function", sm, s0, true, false);
  auto direction = AssociativePlasticFlowDirection("plastic_flow_direction", yield);

  SECTION("model definition")
  {
    // My input should be sufficient for me to evaluate the yield function, hence
    REQUIRE(direction.input() == yield->input());

    REQUIRE(direction.output().has_subaxis("state"));
    REQUIRE(direction.output().subaxis("state").has_variable<SymR2>("plastic_flow_direction"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, direction.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).batch_expand(nbatch);
    in.slice("state").slice("hardening_interface").set(Scalar(200, nbatch), "isotropic_hardening");
    in.slice("state").set(M, "mandel_stress");

    auto exact = direction.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, direction.output(), direction.input());
    finite_differencing_derivative(
        [direction](const LabeledVector & x) { return direction.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
