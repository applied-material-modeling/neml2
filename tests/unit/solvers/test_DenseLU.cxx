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

#include "neml2/neml2.h"
#include "neml2/solvers/DenseLU.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/AssembledMatrix.h"
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/tensors/Tensor.h"

using namespace neml2;

namespace
{
/// Build a single-group AxisLayout for \p n scalar unknowns.
AxisLayout
make_scalar_layout(std::size_t n)
{
  std::vector<VariableName> vars(n);
  std::vector<TensorShape> intmd_shapes(n, TensorShape{});
  std::vector<TensorShape> base_shapes(n, TensorShape{});
  for (std::size_t i = 0; i < n; ++i)
    vars[i] = "u_" + std::to_string(i);
  return AxisLayout({vars}, intmd_shapes, base_shapes, {AxisLayout::IStructure::DENSE});
}
} // namespace

TEST_CASE("DenseLU", "[solvers]")
{
  // Load the solver from the shared input file (same one used by test_NonlinearSolvers.cxx).
  auto factory = load_input("solvers/solvers.i");
  auto solver = factory->get_solver<LinearSolver>("lu");
  REQUIRE(solver != nullptr);

  SECTION("solve Ax = b (2x2 unbatched)")
  {
    // A = [[1, 2], [3, 4]],  b = [5, 11],  exact solution x = [1, 2]
    // Verification: 1*1+2*2=5 and 3*1+4*2=11
    auto layout = make_scalar_layout(2);

    auto A_raw = at::tensor({{1.0, 2.0}, {3.0, 4.0}}).to(at::kDouble);
    auto A_tensor = Tensor(A_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    auto b_raw = at::tensor({5.0, 11.0}).to(at::kDouble);
    auto b_tensor = Tensor(b_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    AssembledMatrix A_mat(layout, layout, {{A_tensor}});
    AssembledVector b_vec(layout, {b_tensor});

    const auto x = solver->solve(A_mat, b_vec);
    REQUIRE(at::allclose(x.tensors[0], at::tensor({1.0, 2.0}).to(at::kDouble)));
  }

  SECTION("solve Ax = b (3x3 unbatched)")
  {
    // A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]],  b = [5, 10, 7],  exact x = [1, 3, 2]
    // Verification: 2+3+0=5, 1+9+2=12 – let me pick a simpler one:
    // A = diag(2, 3, 4), b = [2, 6, 8], exact x = [1, 2, 2]
    auto layout = make_scalar_layout(3);

    auto A_raw = at::tensor({{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 4.0}}).to(at::kDouble);
    auto A_tensor = Tensor(A_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    auto b_raw = at::tensor({2.0, 6.0, 8.0}).to(at::kDouble);
    auto b_tensor = Tensor(b_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    AssembledMatrix A_mat(layout, layout, {{A_tensor}});
    AssembledVector b_vec(layout, {b_tensor});

    const auto x = solver->solve(A_mat, b_vec);
    const auto expected = at::tensor({1.0, 2.0, 2.0}).to(at::kDouble);
    REQUIRE(at::allclose(x.tensors[0], expected));
  }

  SECTION("solve Ax = b (2x2 batched)")
  {
    // Two independent systems in a batch of size 3.
    // For each batch element: A = [[2, 0], [0, 2]], b = [4, 6], exact x = [2, 3]
    auto layout = make_scalar_layout(2);

    const Size batch = 3;
    // A_raw shape: {batch, 2, 2}
    auto A_raw = 2.0 * at::eye(2, at::kDouble).unsqueeze(0).expand({batch, 2, 2});
    auto A_tensor = Tensor(A_raw, /*dynamic_dim=*/1, /*intmd_dim=*/0);

    // b_raw shape: {batch, 2}
    auto b_raw = at::tensor({4.0, 6.0}).to(at::kDouble).unsqueeze(0).expand({batch, 2});
    auto b_tensor = Tensor(b_raw, /*dynamic_dim=*/1, /*intmd_dim=*/0);

    AssembledMatrix A_mat(layout, layout, {{A_tensor}});
    AssembledVector b_vec(layout, {b_tensor});

    const auto x = solver->solve(A_mat, b_vec);

    // Check all batch elements
    const auto expected = at::tensor({2.0, 3.0}).to(at::kDouble).unsqueeze(0).expand({batch, 2});
    REQUIRE(at::allclose(x.tensors[0], expected));
  }

  SECTION("solve AX = B (matrix right-hand side)")
  {
    // A = I, B = [[1, 2], [3, 4]], solution X = B
    auto layout = make_scalar_layout(2);

    auto A_raw = at::eye(2, at::kDouble);
    auto A_tensor = Tensor(A_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    auto B_raw = at::tensor({{1.0, 2.0}, {3.0, 4.0}}).to(at::kDouble);
    auto B_tensor = Tensor(B_raw, /*dynamic_dim=*/0, /*intmd_dim=*/0);

    AssembledMatrix A_mat(layout, layout, {{A_tensor}});
    AssembledMatrix B_mat(layout, layout, {{B_tensor}});

    const auto X = solver->solve(A_mat, B_mat);
    REQUIRE(at::allclose(X.tensors[0][0], B_raw));
  }
}
