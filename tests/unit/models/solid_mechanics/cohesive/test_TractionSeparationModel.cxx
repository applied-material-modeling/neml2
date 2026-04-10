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

#include <cmath>

#include "utils.h"
#include "neml2/neml2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"

using namespace neml2;

// ---------------------------------------------------------------------------
// Helper: build a ValueMap for models whose only non-trivial input is
// forces/displacement_jump (Vec).
// ---------------------------------------------------------------------------
static ValueMap
delta_only_input(const Vec & delta)
{
  return {{VariableName(FORCES, "displacement_jump"), delta}};
}

// ---------------------------------------------------------------------------
// Helper: finite-difference the Jacobian of a model output variable w.r.t.
// a single input variable. Uses the finite_differencing_derivative helper
// from utils.h. Returns a tensor of shape (out_base; in_base).
// ---------------------------------------------------------------------------
static Tensor
fd_jacobian(Model & model,
            ValueMap base_in,
            const VariableName & in_name,
            const VariableName & out_name)
{
  const auto x0 = Tensor(base_in.at(in_name)).clone();
  return finite_differencing_derivative(
      [&](const Tensor & x)
      {
        base_in[in_name] = x;
        return Tensor(model.value(base_in).at(out_name));
      },
      x0);
}

// ===========================================================================
// PureElasticTractionSeparation
// ===========================================================================
TEST_CASE("PureElasticTractionSeparation", "[solid_mechanics/cohesive]")
{
  // Model: T_n = K_n*delta_n, T_si = K_t*delta_si
  // K_n = 100, K_t = 50
  // delta = [0.01, 0.02, 0.005] -> T = [1.0, 1.0, 0.25]
  auto model =
      load_model("models/solid_mechanics/cohesive/PureElasticTractionSeparation.i", "model");

  const auto delta = Vec::fill(0.01, 0.02, 0.005);
  const auto in = delta_only_input(delta);

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    const auto expected = Vec::fill(1.0, 1.0, 0.25);
    REQUIRE(at::allclose(T, expected, /*rtol=*/1e-6, /*atol=*/1e-6));
  }

  SECTION("set_dvalue")
  {
    // Analytic Jacobian is diag(K_n, K_t, K_t) = diag(100, 50, 50)
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    // Finite-difference approximation
    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

// ===========================================================================
// SalehaniIrani3DCTraction
// ===========================================================================
TEST_CASE("SalehaniIrani3DCTraction", "[solid_mechanics/cohesive]")
{
  // Parameters: delta_n0=1, tangential_gap_raw=1 (delta_t0=sqrt(2)),
  //             T_max_n=100, T_max_t=200
  // Test point: delta = [0.5, 0.1, 0.05] (full mixed-mode loading)
  //   b0 = 0.5, b1 = 0.1/sqrt(2), b2 = 0.05/sqrt(2)
  //   x = 0.5 + b1^2 + b2^2 = 0.50625
  //   T_n = e*100 * 0.5 * exp(-0.50625) = 81.922...
  //   T_s1 = sqrt(2e)*200 * b1 * exp(-0.50625) = 19.875...
  //   T_s2 = sqrt(2e)*200 * b2 * exp(-0.50625) = 9.938...
  //
  // NOTE: non-zero ds1 and ds2 are required so that finite-difference
  //       derivatives of the off-diagonal Jacobian components are accurate
  //       (zero inputs trigger O(dx) FD artifacts via the quadratic exponent).
  auto model =
      load_model("models/solid_mechanics/cohesive/SalehaniIrani3DCTraction.i", "model");

  const auto delta = Vec::fill(0.5, 0.1, 0.05);
  const auto in = delta_only_input(delta);

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    // x = 0.5 + (0.1/sqrt(2))^2 + (0.05/sqrt(2))^2 = 0.5 + 0.005 + 0.00125 = 0.50625
    const double x = 0.5 + 0.1 * 0.1 / 2.0 + 0.05 * 0.05 / 2.0;
    const double exp_neg_x = std::exp(-x);
    const double e = std::exp(1.0);
    const double sqrt2 = std::sqrt(2.0);
    const double delta_t0 = sqrt2 * 1.0;
    const double a0 = e * 100.0;
    const double a12 = std::sqrt(2.0 * e) * 200.0;
    const double T_n_expected = a0 * (0.5 / 1.0) * exp_neg_x;
    const double T_s1_expected = a12 * (0.1 / delta_t0) * exp_neg_x;
    const double T_s2_expected = a12 * (0.05 / delta_t0) * exp_neg_x;
    const auto expected = Vec::fill(T_n_expected, T_s1_expected, T_s2_expected);
    REQUIRE(at::allclose(T, expected, /*rtol=*/1e-6, /*atol=*/1e-6));
  }

  SECTION("set_dvalue")
  {
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

// ===========================================================================
// BiLinearMixedModeTraction — elastic regime (no damage)
// ===========================================================================
TEST_CASE("BiLinearMixedModeTraction elastic regime", "[solid_mechanics/cohesive]")
{
  // K=1e4, N=S=100  => delta_init = N/K = 0.01
  // delta=[0.001, 0, 0]: delta_m=0.001 < delta_init=0.01 => d=0
  // T_n = K*delta_n = 1e4*0.001 = 10, T_s = 0
  auto model =
      load_model("models/solid_mechanics/cohesive/BiLinearMixedModeTraction_elastic.i", "model");

  const auto delta = Vec::fill(0.001, 0.0, 0.0);
  const ValueMap in = {
      {VariableName(FORCES, "displacement_jump"), delta},
      {VariableName(OLD_FORCES, "displacement_jump"), delta},
      {VariableName(OLD_STATE, "damage"), Scalar::full(0.0)},
      {VariableName(FORCES, "t"), Scalar::full(1.0)},
      {VariableName(OLD_FORCES, "t"), Scalar::full(0.0)},
  };

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    const auto & d = out.at(VariableName(STATE, "damage"));

    const auto T_expected = Vec::fill(10.0, 0.0, 0.0);
    REQUIRE(at::allclose(T, T_expected, /*rtol=*/1e-6, /*atol=*/1e-6));
    REQUIRE(at::allclose(d, Scalar::full(0.0), /*rtol=*/1e-6, /*atol=*/1e-6));
  }

  SECTION("set_dvalue")
  {
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

// ===========================================================================
// BiLinearMixedModeTraction — damage regime (pure mode I, set_value check)
// ===========================================================================
TEST_CASE("BiLinearMixedModeTraction damage regime set_value", "[solid_mechanics/cohesive]")
{
  // K=1e4, GI_c=1e3, GII_c=2e3, N=S=100, eta=2, viscosity=0
  // delta=[0.02, 0, 0], d_old=0, delta_old=[0.02, 0, 0]
  //   beta=0 (pure mode I), delta_init=0.01, delta_final=20
  //   d = 20*(0.02-0.01)/(0.02*(20-0.01)) = 0.5002501250625313
  //   T_n = (1-d)*K*0.02 = 99.94997498749373
  auto model =
      load_model("models/solid_mechanics/cohesive/BiLinearMixedModeTraction_damage.i", "model");

  const auto delta = Vec::fill(0.02, 0.0, 0.0);
  const ValueMap in = {
      {VariableName(FORCES, "displacement_jump"), delta},
      {VariableName(OLD_FORCES, "displacement_jump"), delta},
      {VariableName(OLD_STATE, "damage"), Scalar::full(0.0)},
      {VariableName(FORCES, "t"), Scalar::full(1.0)},
      {VariableName(OLD_FORCES, "t"), Scalar::full(0.0)},
  };

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    const auto & d = out.at(VariableName(STATE, "damage"));

    const double d_expected = 20.0 * (0.02 - 0.01) / (0.02 * (20.0 - 0.01));
    const double T_n_expected = (1.0 - d_expected) * 1e4 * 0.02;
    REQUIRE(at::allclose(T, Vec::fill(T_n_expected, 0.0, 0.0), /*rtol=*/1e-6, /*atol=*/1e-6));
    REQUIRE(at::allclose(d, Scalar::full(d_expected), /*rtol=*/1e-6, /*atol=*/1e-6));
  }
}

// ===========================================================================
// BiLinearMixedModeTraction — damage regime (mixed mode, set_dvalue check)
// ===========================================================================
TEST_CASE("BiLinearMixedModeTraction damage regime set_dvalue", "[solid_mechanics/cohesive]")
{
  // Use a fully mixed-mode point to avoid degenerate FD behaviour at ds1=ds2=0.
  // K=1e4, GI_c=1e3, GII_c=2e3, N=S=100, eta=2, viscosity=0
  // delta=[0.015, 0.01, 0.005], d_old=0, delta_old=delta_cur (lag_mode_mixity=true)
  //   beta = sqrt(0.01^2+0.005^2)/0.015 = 0.7454
  //   delta_m = sqrt(0.015^2+0.01^2+0.005^2) = 0.01871
  //   -> in damage regime (delta_m > delta_init=0.01)
  auto model =
      load_model("models/solid_mechanics/cohesive/BiLinearMixedModeTraction_damage.i", "model");

  const auto delta = Vec::fill(0.015, 0.01, 0.005);
  const ValueMap in = {
      {VariableName(FORCES, "displacement_jump"), delta},
      {VariableName(OLD_FORCES, "displacement_jump"), delta},
      {VariableName(OLD_STATE, "damage"), Scalar::full(0.0)},
      {VariableName(FORCES, "t"), Scalar::full(1.0)},
      {VariableName(OLD_FORCES, "t"), Scalar::full(0.0)},
  };

  SECTION("set_dvalue")
  {
    // With lag_mode_mixity=true and delta_old = delta_cur, the mode-mixity
    // quantities (delta_init, delta_final) are frozen w.r.t. the current
    // displacement jump. Only the delta_m path contributes to dT/d(delta).
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

// ===========================================================================
// ExpTractionSeparation — reversible (no damage history)
// ===========================================================================
TEST_CASE("ExpTractionSeparation reversible", "[solid_mechanics/cohesive]")
{
  // Parameters: Gc=1.0, delta0=0.5, beta=0.5, irreversible_damage=false
  // delta = [0.3, 0.2, 0.1], old_max = 0.0
  //   delta_eff = sqrt(0.3^2 + 0.5*(0.2^2+0.1^2)) = sqrt(0.09+0.025) = sqrt(0.115)
  //   one_m_d   = exp(-sqrt(0.115)/0.5)
  //   c         = 1.0 / 0.5^2 = 4.0
  //   T_i       = one_m_d * 4.0 * delta_i
  auto model = load_model(
      "models/solid_mechanics/cohesive/ExpTractionSeparation_reversible.i", "model");

  const ValueMap in = {
      {VariableName(FORCES, "displacement_jump"), Vec::fill(0.3, 0.2, 0.1)},
      {VariableName(OLD_STATE, "effective_displacement_jump_scalar_max"), Scalar::full(0.0)},
  };

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    const auto & delta_eff_max =
        out.at(VariableName(STATE, "effective_displacement_jump_scalar_max"));

    const double delta_eff = std::sqrt(0.115);
    const double one_m_d = std::exp(-delta_eff / 0.5);
    const double c = 4.0;
    const auto T_expected = Vec::fill(one_m_d * c * 0.3, one_m_d * c * 0.2, one_m_d * c * 0.1);

    REQUIRE(at::allclose(T, T_expected, /*rtol=*/1e-6, /*atol=*/1e-6));
    REQUIRE(at::allclose(
        delta_eff_max, Scalar::full(delta_eff), /*rtol=*/1e-6, /*atol=*/1e-6));
  }

  SECTION("set_dvalue")
  {
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

// ===========================================================================
// ExpTractionSeparation — irreversible, clamped by old_max
// ===========================================================================
TEST_CASE("ExpTractionSeparation irreversible clamped", "[solid_mechanics/cohesive]")
{
  // Parameters: Gc=1.0, delta0=0.5, beta=0.5, irreversible_damage=true
  // delta = [0.3, 0.2, 0.1], old_max = 0.5
  //   delta_eff_raw = sqrt(0.115) ~ 0.33912 < 0.5 => delta_eff clamped to 0.5
  //   one_m_d = exp(-0.5/0.5) = exp(-1)
  //   c = 4.0
  //   T_i = exp(-1) * 4.0 * delta_i
  //   effective_displacement_jump_scalar_max = 0.5 (unchanged)
  // When clamped, dT/d(delta) = one_m_d * c * I (delta_eff is constant w.r.t. delta)
  auto model = load_model(
      "models/solid_mechanics/cohesive/ExpTractionSeparation_irreversible.i", "model");

  const ValueMap in = {
      {VariableName(FORCES, "displacement_jump"), Vec::fill(0.3, 0.2, 0.1)},
      {VariableName(OLD_STATE, "effective_displacement_jump_scalar_max"), Scalar::full(0.5)},
  };

  SECTION("set_value")
  {
    const auto out = model->value(in);
    const auto & T = out.at(VariableName(STATE, "traction"));
    const auto & delta_eff_max =
        out.at(VariableName(STATE, "effective_displacement_jump_scalar_max"));

    const double one_m_d = std::exp(-1.0);
    const double c = 4.0;
    const auto T_expected = Vec::fill(one_m_d * c * 0.3, one_m_d * c * 0.2, one_m_d * c * 0.1);

    REQUIRE(at::allclose(T, T_expected, /*rtol=*/1e-6, /*atol=*/1e-6));
    REQUIRE(at::allclose(
        delta_eff_max, Scalar::full(0.5), /*rtol=*/1e-6, /*atol=*/1e-6));
  }

  SECTION("set_dvalue")
  {
    // Clamped regime: delta_eff is constant w.r.t. delta, so
    // dT/d(delta) = one_m_d * c * I (a scaled identity).
    const auto [out, dout] = model->value_and_dvalue(in);

    const auto & dT_ddelta =
        dout.at(VariableName(STATE, "traction")).at(VariableName(FORCES, "displacement_jump"));

    const auto dT_ddelta_fd =
        fd_jacobian(*model, in, {FORCES, "displacement_jump"}, {STATE, "traction"});

    REQUIRE(at::allclose(dT_ddelta, dT_ddelta_fd, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}
