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

#include "neml2/neml2.h"
#include "neml2/solvers/NonlinearSolver.h"

using namespace neml2;

TEST_CASE("NonlinearSolver", "[solvers]")
{
  // System shape
  TensorShape batch_sz = {2};
  std::size_t n = 4;

  // Create the solver
  auto factory = load_input("solvers/solvers.i");
  auto solver_name = GENERATE("newton", "newton_with_line_search");

  SECTION(solver_name)
  {
    auto solver = factory->get_solver<NonlinearSolver>(solver_name);

    SECTION("power")
    {
      // Initial guess
      std::vector<TensorShape> x_shapes(n, TensorShape{});
      std::vector<Tensor> x_data(n, Scalar::full(batch_sz, {}, 2.0));
      HVector x(x_data, x_shapes);

      // Create the nonlinear system
      PowerTestSystem system;

      // Solve
      auto res = solver->solve(system, x);
      REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

      // Check solution
      const auto expected = system.exact_solution(x);
      for (std::size_t i = 0; i < n; i++)
        REQUIRE(at::allclose(res.solution[i], expected[i]));
    }

    SECTION("Rosenbrock")
    {
      // Initial guess
      std::vector<TensorShape> x_shapes(n, TensorShape{});
      std::vector<Tensor> x_data(n, Scalar::full(batch_sz, {}, 0.75));
      HVector x(x_data, x_shapes);

      // Create the nonlinear system
      RosenbrockTestSystem system;

      // Solve
      auto res = solver->solve(system, x);
      REQUIRE(res.ret == NonlinearSolver::RetCode::SUCCESS);

      // Check solution
      const auto expected = system.exact_solution(x);
      for (std::size_t i = 0; i < n; i++)
        REQUIRE(at::allclose(res.solution[i], expected[i]));
    }
  }
}
