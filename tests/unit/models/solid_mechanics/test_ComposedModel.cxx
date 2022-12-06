#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "models/ComposedModel.h"
#include "models/solid_mechanics/ElasticStrain.h"
#include "models/solid_mechanics/LinearIsotropicElasticity.h"
#include "models/solid_mechanics/NoKinematicHardening.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"
#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"
#include "models/solid_mechanics/PlasticStrainRate.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "models/IdentityMap.h"
#include "models/forces/QuasiStaticForce.h"
#include "solvers/NewtonNonlinearSolver.h"

TEST_CASE("Linear DAG", "[ComposedModel]")
{
  TorchSize nbatch = 10;
  Scalar E(100, nbatch);
  Scalar nu(0.3, nbatch);
  auto estrain = ElasticStrain("elastic_strain");
  auto elasticity = LinearIsotropicElasticity<false>("elasticity", E, nu);
  auto kinharden = NoKinematicHardening("kinematic_hardening");

  // inputs --> "elastic_strain" --> "elasticity" --> "kinharden" --> outputs
  // inputs: total strain, plastic strain
  // outputs: mandel stress
  auto model = ComposedModel("foo");
  model.registerModel(estrain);
  model.registerModel(elasticity);
  model.registerModel(kinharden);
  model.registerDependency("elastic_strain", "elasticity");
  model.registerDependency("elasticity", "kinematic_hardening");

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("forces"));
    REQUIRE(model.input().subaxis("forces").has_variable<SymR2>("total_strain"));
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("plastic_strain"));
    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("mandel_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto E = SymR2::init(0.1, 0.05, 0).expand_batch(nbatch);
    auto Ep = SymR2::init(0.01, 0.01, 0).expand_batch(nbatch);
    in.slice(0, "forces").set(E, "total_strain");
    in.slice(0, "state").set(Ep, "plastic_strain");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("Y-junction DAG", "[ComposedModel]")
{
  TorchSize nbatch = 10;
  Scalar s0 = 100.0;
  Scalar K = 1000.0;
  auto isoharden = LinearIsotropicHardening("isotropic_hardening", s0, K);
  auto kinharden = NoKinematicHardening("kinematic_hardening");
  auto yield = J2IsotropicYieldFunction("yield_function");

  // inputs --> "isotropic_hardening" --
  //                                    `
  //                                     `--> "yield_function" --> outputs
  //                                     '
  //                                    '
  // inputs -> "kinematic_hardening" ---
  //
  // inputs: cauchy stress, equivalent plastic strain
  // outputs: yield function
  auto model = ComposedModel("foo");
  model.registerModel(isoharden);
  model.registerModel(kinharden);
  model.registerModel(yield);
  model.registerDependency("isotropic_hardening", "yield_function");
  model.registerDependency("kinematic_hardening", "yield_function");

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("equivalent_plastic_strain"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("yield_function"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto S = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(nbatch);
    in.slice(0, "state").set(S, "cauchy_stress");
    in.slice(0, "state").set(Scalar(0.1, nbatch), "equivalent_plastic_strain");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("diamond pattern", "[ComposedModel]")
{
  TorchSize nbatch = 10;
  Scalar eta = 150;
  Scalar n = 6;
  auto yield = J2IsotropicYieldFunction("yield_function");
  auto direction = AssociativePlasticFlowDirection("plastic_flow_direction", yield);
  auto eprate = PerzynaPlasticFlowRate("plastic_flow_rate", eta, n);
  auto Eprate = PlasticStrainRate("plastic_strain_rate");

  auto model = ComposedModel("foo");
  model.registerModel(yield);
  model.registerModel(direction);
  model.registerModel(eprate);
  model.registerModel(Eprate);
  model.registerDependency("yield_function", "plastic_flow_rate");
  model.registerDependency("plastic_flow_rate", "plastic_strain_rate");
  model.registerDependency("plastic_flow_direction", "plastic_strain_rate");

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("state"));
    REQUIRE(model.input().subaxis("state").has_variable<SymR2>("mandel_stress"));
    REQUIRE(model.input().subaxis("state").has_variable<Scalar>("isotropic_hardening"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("plastic_strain_rate"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, model.input());
    auto M = SymR2::init(100, 110, 100, 100, 100, 100).expand_batch(nbatch);
    in.slice(0, "state").set(M, "mandel_stress");
    in.slice(0, "state").set(Scalar(200, nbatch), "isotropic_hardening");

    auto exact = model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
