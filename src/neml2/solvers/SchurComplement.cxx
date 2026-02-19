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

#include "neml2/solvers/SchurComplement.h"
#include "neml2/tensors/Tensor.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/functions/linalg/solve.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/mv.h"
#include "neml2/equation_systems/LinearSystem.h"
#include "neml2/equation_systems/NonlinearSystem.h"
#include "neml2/equation_systems/SparseTensorList.h"
#include "neml2/equation_systems/assembly.h"

namespace neml2
{
namespace
{
template <typename T, typename Getter>
std::vector<T>
flatten_grouped(std::size_t num_groups, Getter && getter)
{
  std::vector<T> flattened;
  for (std::size_t i = 0; i < num_groups; ++i)
    for (const auto & item : getter(i))
      flattened.push_back(item);
  return flattened;
}
}

register_NEML2_object(SchurComplement);

OptionSet
SchurComplement::expected_options()
{
  OptionSet options = LinearSolver::expected_options();

  options.doc() =
      "Schur complement linear solver. This solver partitions the linear system "
      "into primary and Schur variable groups and applies the Schur complement "
      "method. The system is partitioned as [[A11, A12], [A21, A22]] with "
      "unknowns [u1, u2]. The Schur complement S = A22 - A21 * A11^(-1) * A12 "
      "is formed, then u2 is solved from S * u2 = b2 - A21 * A11^(-1) * b1, "
      "and u1 is back-solved from A11 * u1 = b1 - A12 * u2.";

  options.set<unsigned int>("primary_group") = 0;
  options.set("primary_group").doc() =
      "Index of the variable group to use as primary unknowns (solved first via back-substitution).";

  options.set<std::string>("primary_linear_solver") = "";
  options.set("primary_linear_solver").doc() =
      "Linear solver for the primary block A11. Currently uses dense LU decomposition.";

  options.set<unsigned int>("schur_group") = 1;
  options.set("schur_group").doc() =
      "Index of the variable group for the Schur complement (condensed unknowns).";

  options.set<std::string>("schur_linear_solver") = "";
  options.set("schur_linear_solver").doc() =
      "Linear solver for the Schur complement block. Currently uses dense LU decomposition.";

  return options;
}

SchurComplement::SchurComplement(const OptionSet & options)
  : LinearSolver(options),
    _primary_group(options.get<unsigned int>("primary_group")),
    _schur_group(options.get<unsigned int>("schur_group"))
{
  // Validate group indices
  if (_primary_group == _schur_group)
    throw NEMLException("primary_group and schur_group must be different.");
}

void
SchurComplement::setup()
{
  LinearSolver::setup();

  // Get sub-solvers if specified (for validation)
  const auto & primary_solver_name = input_options().get<std::string>("primary_linear_solver");
  const auto & schur_solver_name = input_options().get<std::string>("schur_linear_solver");

  if (!primary_solver_name.empty())
    _primary_solver = get_solver<LinearSolver>("primary_linear_solver");

  if (!schur_solver_name.empty())
    _schur_solver = get_solver<LinearSolver>("schur_linear_solver");
}

Tensor
SchurComplement::extract_block(const Tensor & A,
                               Size row_start,
                               Size row_size,
                               Size col_start,
                               Size col_size)
{
  // A is expected to have base shape (m, n)
  // Extract the block A[row_start:row_start+row_size, col_start:col_start+col_size]
  using namespace at::indexing;
  auto block = A.base_index({Slice(row_start, row_start + row_size),
                             Slice(col_start, col_start + col_size)});
  return block;
}

Tensor
SchurComplement::extract_subvector(const Tensor & b, Size start, Size size)
{
  // b is expected to have base shape (m,)
  // Extract the subvector b[start:start+size]
  using namespace at::indexing;
  auto subvec = b.base_index({Slice(start, start + size)});
  return subvec;
}

Size
SchurComplement::compute_group_size(const std::vector<TensorShape> & intmd_shapes,
                                    const std::vector<TensorShape> & base_shapes)
{
  Size total = 0;
  for (std::size_t i = 0; i < base_shapes.size(); ++i)
  {
    Size var_size = utils::numel(base_shapes[i]);
    if (i < intmd_shapes.size())
      var_size *= utils::numel(intmd_shapes[i]);
    total += var_size;
  }
  return total;
}

SparseTensorList
SchurComplement::solve(LinearSystem & sys) const
{
  if (sys.n_ugroup() != 2 || sys.n_bgroup() != 2)
    throw NEMLException("SchurComplement solver currently requires exactly 2 variable groups. "
                        "Found " + std::to_string(sys.n_ugroup()) + " unknown groups and " +
                        std::to_string(sys.n_bgroup()) + " residual groups.");

  if (_primary_group >= sys.n_ugroup() || _schur_group >= sys.n_ugroup())
    throw NEMLException("primary_group and schur_group must be valid group indices in [0, " +
                        std::to_string(sys.n_ugroup() - 1) + "].");

  // Get layouts for assembly
  const auto bilayout = flatten_grouped<TensorShape>(
      sys.n_bgroup(), [&](std::size_t i) -> const auto & { return sys.intmd_blayout(i); });
  const auto uilayout = flatten_grouped<TensorShape>(
      sys.n_ugroup(), [&](std::size_t i) -> const auto & { return sys.intmd_ulayout(i); });
  const auto blayout = flatten_grouped<TensorShape>(
      sys.n_bgroup(), [&](std::size_t i) -> const auto & { return sys.blayout(i); });
  const auto ulayout = flatten_grouped<TensorShape>(
      sys.n_ugroup(), [&](std::size_t i) -> const auto & { return sys.ulayout(i); });

  // Assemble A and b into flat tensors
  const auto [A, b] = sys.A_and_b();
  const auto Af = assemble(A, bilayout, uilayout, blayout, ulayout);
  const auto bf = assemble(b, bilayout, blayout);

  // Compute group sizes
  // The unknowns are ordered by group, so we can compute offsets
  const auto & group0_intmd_ulayout = sys.intmd_ulayout(0);
  const auto & group0_ulayout = sys.ulayout(0);
  const auto & group1_intmd_ulayout = sys.intmd_ulayout(1);
  const auto & group1_ulayout = sys.ulayout(1);

  Size size0 = compute_group_size(group0_intmd_ulayout, group0_ulayout);
  Size size1 = compute_group_size(group1_intmd_ulayout, group1_ulayout);

  // Determine which group is primary and which is Schur
  Size primary_size, schur_size, primary_start, schur_start;
  if (_primary_group == 0)
  {
    primary_start = 0;
    primary_size = size0;
    schur_start = size0;
    schur_size = size1;
  }
  else
  {
    primary_start = size0;
    primary_size = size1;
    schur_start = 0;
    schur_size = size0;
  }

  // Extract sub-blocks from the matrix A
  // A = [[A11, A12], [A21, A22]] where:
  //   A11 = primary-primary block
  //   A12 = primary-schur block
  //   A21 = schur-primary block
  //   A22 = schur-schur block
  const auto A11 = extract_block(Af, primary_start, primary_size, primary_start, primary_size);
  const auto A12 = extract_block(Af, primary_start, primary_size, schur_start, schur_size);
  const auto A21 = extract_block(Af, schur_start, schur_size, primary_start, primary_size);
  const auto A22 = extract_block(Af, schur_start, schur_size, schur_start, schur_size);

  // Extract sub-vectors from b
  const auto b1 = extract_subvector(bf, primary_start, primary_size);
  const auto b2 = extract_subvector(bf, schur_start, schur_size);

  // Step 1: Compute A11^{-1} * A12 and A11^{-1} * b1
  // We solve A11 * X = A12 and A11 * y = b1
  const auto A11_inv_A12 = linalg::solve(A11, A12);
  const auto A11_inv_b1 = linalg::solve(A11, b1);

  // Step 2: Form the Schur complement S = A22 - A21 * A11^{-1} * A12
  const auto S = A22 - mm(A21, A11_inv_A12);

  // Step 3: Form the modified RHS: rhs_schur = b2 - A21 * A11^{-1} * b1
  const auto rhs_schur = b2 - mv(A21, A11_inv_b1);

  // Step 4: Solve for the Schur unknowns: S * u2 = rhs_schur
  const auto u_schur = linalg::solve(S, rhs_schur);

  // Step 5: Back-solve for the primary unknowns: A11 * u1 = b1 - A12 * u2
  const auto rhs_primary = b1 - mv(A12, u_schur);
  const auto u_primary = linalg::solve(A11, rhs_primary);

  // Step 6: Reassemble the solution in the correct order
  // The solution ordering matches the system ordering (group 0 first, then group 1)
  Tensor xf;
  if (_primary_group == 0)
    xf = base_cat({u_primary, u_schur}, -1);
  else
    xf = base_cat({u_schur, u_primary}, -1);

  // Disassemble the solution
  return disassemble(xf, uilayout, ulayout);
}

SparseTensorList
SchurComplement::ift(NonlinearSystem & sys) const
{
  if (sys.n_ugroup() != 2 || sys.n_bgroup() != 2)
    throw NEMLException("SchurComplement solver currently requires exactly 2 variable groups. "
                        "Found " + std::to_string(sys.n_ugroup()) + " unknown groups and " +
                        std::to_string(sys.n_bgroup()) + " residual groups.");

  if (_primary_group >= sys.n_ugroup() || _schur_group >= sys.n_ugroup())
    throw NEMLException("primary_group and schur_group must be valid group indices in [0, " +
                        std::to_string(sys.n_ugroup() - 1) + "].");

  // Get layouts
  const auto bilayout = flatten_grouped<TensorShape>(
      sys.n_bgroup(), [&](std::size_t i) -> const auto & { return sys.intmd_blayout(i); });
  const auto uilayout = flatten_grouped<TensorShape>(
      sys.n_ugroup(), [&](std::size_t i) -> const auto & { return sys.intmd_ulayout(i); });
  const auto gilayout = sys.intmd_glayout();
  const auto blayout = flatten_grouped<TensorShape>(
      sys.n_bgroup(), [&](std::size_t i) -> const auto & { return sys.blayout(i); });
  const auto ulayout = flatten_grouped<TensorShape>(
      sys.n_ugroup(), [&](std::size_t i) -> const auto & { return sys.ulayout(i); });
  const auto glayout = sys.glayout();

  // Assemble A and B into flat tensors
  const auto [A, B] = sys.A_and_B();
  const auto Af = assemble(A, bilayout, uilayout, blayout, ulayout);
  const auto Bf = assemble(B, bilayout, gilayout, blayout, glayout);

  // Compute group sizes
  const auto & group0_intmd_ulayout = sys.intmd_ulayout(0);
  const auto & group0_ulayout = sys.ulayout(0);
  const auto & group1_intmd_ulayout = sys.intmd_ulayout(1);
  const auto & group1_ulayout = sys.ulayout(1);

  Size size0 = compute_group_size(group0_intmd_ulayout, group0_ulayout);
  Size size1 = compute_group_size(group1_intmd_ulayout, group1_ulayout);

  // Determine which group is primary and which is Schur
  Size primary_size, schur_size, primary_start, schur_start;
  if (_primary_group == 0)
  {
    primary_start = 0;
    primary_size = size0;
    schur_start = size0;
    schur_size = size1;
  }
  else
  {
    primary_start = size0;
    primary_size = size1;
    schur_start = 0;
    schur_size = size0;
  }

  // Extract sub-blocks from A
  const auto A11 = extract_block(Af, primary_start, primary_size, primary_start, primary_size);
  const auto A12 = extract_block(Af, primary_start, primary_size, schur_start, schur_size);
  const auto A21 = extract_block(Af, schur_start, schur_size, primary_start, primary_size);
  const auto A22 = extract_block(Af, schur_start, schur_size, schur_start, schur_size);

  // Extract sub-blocks from B (rows correspond to unknowns)
  const Size p = Bf.base_size(-1); // number of given variables
  const auto B1 = extract_block(Bf, primary_start, primary_size, 0, p);
  const auto B2 = extract_block(Bf, schur_start, schur_size, 0, p);

  // Step 1: Compute A11^{-1} * A12 and A11^{-1} * B1
  const auto A11_inv_A12 = linalg::solve(A11, A12);
  const auto A11_inv_B1 = linalg::solve(A11, B1);

  // Step 2: Form the Schur complement S = A22 - A21 * A11^{-1} * A12
  const auto S = A22 - mm(A21, A11_inv_A12);

  // Step 3: Form the modified RHS: RHS_schur = B2 - A21 * A11^{-1} * B1
  const auto RHS_schur = B2 - mm(A21, A11_inv_B1);

  // Step 4: Solve for the Schur derivative: S * X2 = RHS_schur
  const auto X_schur = linalg::solve(S, RHS_schur);

  // Step 5: Back-solve for the primary derivative: A11 * X1 = B1 - A12 * X2
  const auto RHS_primary = B1 - mm(A12, X_schur);
  const auto X_primary = linalg::solve(A11, RHS_primary);

  // Step 6: Reassemble the solution in the correct order
  Tensor Xf;
  if (_primary_group == 0)
    Xf = base_cat({X_primary, X_schur}, -2);
  else
    Xf = base_cat({X_schur, X_primary}, -2);

  // The result is du/dg, but IFT expects -du/dg
  Xf = -Xf;

  // Disassemble the solution
  return disassemble(Xf, uilayout, gilayout, ulayout, glayout);
}

} // namespace neml2
