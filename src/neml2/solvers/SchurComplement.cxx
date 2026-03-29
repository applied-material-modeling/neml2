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
#include "neml2/equation_systems/AssembledMatrix.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(SchurComplement);

OptionSet
SchurComplement::expected_options()
{
  OptionSet options = LinearSolver::expected_options();
  options.doc() =
      "Schur complement linear solver. Solves a block-partitioned system A x = b by forming and "
      "solving the Schur complement of the primary block.";

  options.set<unsigned int>("residual_primary_group") = 0;
  options.set("residual_primary_group").doc() =
      "Row (residual) group index of the primary block. The system must have exactly 2 residual "
      "groups; the other group is automatically the Schur complement residual group.";

  options.set<unsigned int>("unknown_primary_group") = 0;
  options.set("unknown_primary_group").doc() =
      "Column (unknown) group index of the primary block. The system must have exactly 2 unknown "
      "groups; the other group is automatically the Schur complement unknown group.";

  options.set<std::string>("primary_solver");
  options.set("primary_solver").doc() = "Linear solver used for the primary block A_pp";

  options.set<std::string>("schur_solver");
  options.set("schur_solver").doc() = "Linear solver used for the Schur complement block S";

  return options;
}

SchurComplement::SchurComplement(const OptionSet & options)
  : LinearSolver(options),
    _rp(options.get<unsigned int>("residual_primary_group")),
    _rs(1 - options.get<unsigned int>("residual_primary_group")),
    _up(options.get<unsigned int>("unknown_primary_group")),
    _us(1 - options.get<unsigned int>("unknown_primary_group"))
{
  _primary_solver = get_solver<LinearSolver>("primary_solver");
  _schur_solver = get_solver<LinearSolver>("schur_solver");
}

AssembledVector
SchurComplement::solve(const AssembledMatrix & A, const AssembledVector & b) const
{
  neml_assert_dbg(A.row_layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 row groups in A, got ",
                  A.row_layout.ngroup());
  neml_assert_dbg(A.col_layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 column groups in A, got ",
                  A.col_layout.ngroup());
  neml_assert_dbg(b.layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 vector groups in b, got ",
                  b.layout.ngroup());

  // Extract the four blocks using view layouts
  const auto A_pp = A.group(_rp, _up);
  const auto A_ps = A.group(_rp, _us);
  const auto A_sp = A.group(_rs, _up);
  const auto A_ss = A.group(_rs, _us);

  // Extract the two b groups
  const auto b_p = b.group(_rp);
  const auto b_s = b.group(_rs);

  // Step 1: Y = A_pp^{-1} A_ps
  const auto Y = _primary_solver->solve(A_pp, A_ps);

  // Step 2: z = A_pp^{-1} b_p
  const auto z = _primary_solver->solve(A_pp, b_p);

  // Step 3: S = A_ss - A_sp Y
  const auto S = A_ss - A_sp * Y;

  // Step 4: d = b_s - A_sp z
  const auto d = b_s - A_sp * z;

  // Step 5: x_s = S^{-1} d
  const auto x_s = _schur_solver->solve(S, d);

  // Step 6: x_p = z - Y x_s
  const auto x_p = z - Y * x_s;

  // Assemble the full solution
  AssembledVector x(A.col_layout);
  const auto [pp_s, pp_e] = A.col_layout.group_offsets(_up);
  const auto [ss_s, ss_e] = A.col_layout.group_offsets(_us);
  for (std::size_t i = 0; i < pp_e - pp_s; ++i)
    x.tensors[pp_s + i] = x_p.tensors[i];
  for (std::size_t i = 0; i < ss_e - ss_s; ++i)
    x.tensors[ss_s + i] = x_s.tensors[i];

  return x;
}

AssembledMatrix
SchurComplement::solve(const AssembledMatrix & A, const AssembledMatrix & B) const
{
  neml_assert_dbg(A.row_layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 row groups in A, got ",
                  A.row_layout.ngroup());
  neml_assert_dbg(A.col_layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 column groups in A, got ",
                  A.col_layout.ngroup());
  neml_assert_dbg(B.row_layout.ngroup() == 2,
                  "SchurComplement requires exactly 2 row groups in B, got ",
                  B.row_layout.ngroup());
  neml_assert_dbg(B.col_layout.ngroup() == 1,
                  "SchurComplement requires exactly 1 column group in B, got ",
                  B.col_layout.ngroup());

  // Extract the four blocks using view layouts
  const auto A_pp = A.group(_rp, _up);
  const auto A_ps = A.group(_rp, _us);
  const auto A_sp = A.group(_rs, _up);
  const auto A_ss = A.group(_rs, _us);

  // Extract the two row groups of B
  const auto B_p = B.group(_rp, 0);
  const auto B_s = B.group(_rs, 0);

  // Step 1: Y = A_pp^{-1} A_ps
  const auto Y = _primary_solver->solve(A_pp, A_ps);

  // Step 2: Z = A_pp^{-1} B_p
  const auto Z = _primary_solver->solve(A_pp, B_p);

  // Step 3: S = A_ss - A_sp Y
  const auto S = A_ss - A_sp * Y;

  // Step 4: D = B_s - A_sp Z
  const auto D = B_s - A_sp * Z;

  // Step 5: X_s = S^{-1} D
  const auto X_s = _schur_solver->solve(S, D);

  // Step 6: X_p = Z - Y X_s
  const auto X_p = Z - Y * X_s;

  // Assemble the full solution
  AssembledMatrix X(A.col_layout, B.col_layout);
  const auto [pp_s, pp_e] = A.col_layout.group_offsets(_up);
  const auto [ss_s, ss_e] = A.col_layout.group_offsets(_us);
  for (std::size_t i = 0; i < pp_e - pp_s; ++i)
    X.tensors[pp_s + i] = X_p.tensors[i];
  for (std::size_t i = 0; i < ss_e - ss_s; ++i)
    X.tensors[ss_s + i] = X_s.tensors[i];

  return X;
}

} // namespace neml2
