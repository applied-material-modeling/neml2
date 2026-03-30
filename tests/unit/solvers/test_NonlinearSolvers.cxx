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
#include <catch2/generators/catch_generators.hpp>

#include "SampleNonlinearSystems.h"

#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/neml2.h"
#include "neml2/solvers/NonlinearSolver.h"

using namespace neml2;

TEST_CASE("NonlinearSolver", "[solvers]")
{
  // System shape
  TensorShape batch_sz = {2};
  Size n = 4;

  // Create the solver
  auto factory = load_input("solvers/solvers.i");
  auto solver_name = GENERATE("newton", "newton_with_line_search");

  SECTION(solver_name)
  {
    auto solver = factory->get_solver<NonlinearSolver>(solver_name);

    SECTION("power")
    {
      // Create the nonlinear system
      PowerTestSystem eq_sys(batch_sz, n);
      eq_sys.init();

      // Initial guess
      AssembledVector u0(eq_sys.ulayout()->view(), {Tensor::full(batch_sz, {}, n, 2.0)});
      eq_sys.set_u(u0);

      // Solve
      auto res = solver->solve(eq_sys);
      REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

      // Check solution
      const auto sol = eq_sys.u();
      const auto expected = eq_sys.exact_solution(u0);
      REQUIRE(at::allclose(sol.tensors[0], expected.tensors[0]));
    }

    SECTION("Rosenbrock")
    {
      // Create the nonlinear system
      RosenbrockTestSystem eq_sys(batch_sz, n);
      eq_sys.init();

      // Initial guess
      AssembledVector u0(eq_sys.ulayout()->view(), {Tensor::full(batch_sz, {}, n, 0.75)});
      eq_sys.set_u(u0);

      // Solve
      auto res = solver->solve(eq_sys);
      REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

      // Check solution
      const auto sol = eq_sys.u();
      const auto expected = eq_sys.exact_solution(u0);
      REQUIRE(at::allclose(sol.tensors[0], expected.tensors[0]));
    }
  }
}

TEST_CASE("NonlinearSolver/SchurComplement", "[solvers]")
{
  // System shape: 8 DOFs split into 2 groups of 5 and 3
  // primary = {u_0, u_1, u_2, u_3, u_4},
  // Schur = {u_5, u_6, u_7}
  TensorShape batch_sz = {2};
  Size n = 8;
  std::vector<Size> group_sizes = {5, 3};

  auto factory = load_input("solvers/solvers.i");
  auto solver = factory->get_solver<NonlinearSolver>("newton_sc");

  SECTION("power")
  {
    PowerTestSystem eq_sys(batch_sz, n, group_sizes, group_sizes);
    eq_sys.init();

    AssembledVector u0(eq_sys.ulayout()->view(),
                       {Tensor::full(batch_sz, {}, 5, 2.0), Tensor::full(batch_sz, {}, 3, 2.0)});
    eq_sys.set_u(u0);

    auto res = solver->solve(eq_sys);
    REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

    const auto sol = eq_sys.u();
    const auto expected = eq_sys.exact_solution(u0);
    REQUIRE(at::allclose(sol.tensors[0], expected.tensors[0]));
    REQUIRE(at::allclose(sol.tensors[1], expected.tensors[1]));
  }

  SECTION("Rosenbrock")
  {
    RosenbrockTestSystem eq_sys(batch_sz, n, group_sizes, group_sizes);
    eq_sys.init();

    AssembledVector u0(eq_sys.ulayout()->view(),
                       {Tensor::full(batch_sz, {}, 5, 0.75), Tensor::full(batch_sz, {}, 3, 0.75)});
    eq_sys.set_u(u0);

    auto res = solver->solve(eq_sys);
    REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

    const auto sol = eq_sys.u();
    const auto expected = eq_sys.exact_solution(u0);
    REQUIRE(at::allclose(sol.tensors[0], expected.tensors[0]));
    REQUIRE(at::allclose(sol.tensors[1], expected.tensors[1]));
  }
}
