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
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/SparseVector.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/mv.h"

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

  options.set<unsigned int>("primary_group") = 0;
  options.set("primary_group").doc() = "Row/column group index of the primary (pivot) block";

  options.set<unsigned int>("schur_group") = 1;
  options.set("schur_group").doc() = "Row/column group index of the Schur complement block";

  options.set<std::string>("primary_solver");
  options.set("primary_solver").doc() = "Linear solver used for the primary block A_pp";

  options.set<std::string>("schur_solver");
  options.set("schur_solver").doc() = "Linear solver used for the Schur complement block S";

  return options;
}

SchurComplement::SchurComplement(const OptionSet & options)
  : LinearSolver(options),
    _primary_group(options.get<unsigned int>("primary_group")),
    _schur_group(options.get<unsigned int>("schur_group"))
{
  _primary_solver = get_solver<LinearSolver>("primary_solver");
  _schur_solver = get_solver<LinearSolver>("schur_solver");
}

SparseVector
SchurComplement::solve(const SparseMatrix & A, const SparseVector & b) const
{
  neml_assert_dbg(
      A.row_ngroup() >= 2, "SchurComplement requires at least 2 row groups, got ", A.row_ngroup());
  neml_assert_dbg(A.col_ngroup() >= 2,
                  "SchurComplement requires at least 2 column groups, got ",
                  A.col_ngroup());

  // Extract the four blocks using view layouts
  const auto A_pp = A.group(_primary_group, _primary_group);
  const auto A_ps = A.group(_primary_group, _schur_group);
  const auto A_sp = A.group(_schur_group, _primary_group);
  const auto A_ss = A.group(_schur_group, _schur_group);

  // Extract the two b groups by slicing tensors
  const auto [p_s, p_e] = A.row_layout.group_offsets(_primary_group);
  const auto [s_s, s_e] = A.row_layout.group_offsets(_schur_group);
  const SparseVector b_p(
      A.row_layout.group(_primary_group),
      std::vector<Tensor>(b.tensors.begin() + Size(p_s), b.tensors.begin() + Size(p_e)));
  const SparseVector b_s(
      A.row_layout.group(_schur_group),
      std::vector<Tensor>(b.tensors.begin() + Size(s_s), b.tensors.begin() + Size(s_e)));

  // Step 1: Y = A_pp^{-1} A_ps
  const auto Y = _primary_solver->solve(A_pp, A_ps);

  // Step 2: z = A_pp^{-1} b_p
  const auto z = _primary_solver->solve(A_pp, b_p);

  // Step 3: S = A_ss - A_sp Y
  const auto A_sp_f = A_sp.assemble(/*assemble_intmd=*/false);
  const auto Y_f = Y.assemble(/*assemble_intmd=*/false);
  const auto S_f = A_ss.assemble(/*assemble_intmd=*/false) - mm(A_sp_f, Y_f);

  // Step 4: d = b_s - A_sp z
  const auto z_f = z.assemble(/*assemble_intmd=*/false);
  const auto d_f = b_s.assemble(/*assemble_intmd=*/false) - mv(A_sp_f, z_f);

  // Wrap S and d for the schur_solver
  const auto schur_col = A.col_layout.group(_schur_group);
  SparseMatrix S_mat(schur_col, schur_col);
  S_mat.disassemble(S_f, /*assemble_intmd=*/false);
  SparseVector d_mat(schur_col);
  d_mat.disassemble(d_f, /*assemble_intmd=*/false);

  // Step 5: x_s = S^{-1} d
  const auto x_s = _schur_solver->solve(S_mat, d_mat);

  // Step 6: x_p = z - Y x_s
  const auto x_p_f = z_f - mv(Y_f, x_s.assemble(/*assemble_intmd=*/false));
  const auto primary_col = A.col_layout.group(_primary_group);
  SparseVector x_p(primary_col);
  x_p.disassemble(x_p_f, /*assemble_intmd=*/false);

  // Assemble the full solution
  SparseVector x(A.col_layout);
  const auto [pp_s, pp_e] = A.col_layout.group_offsets(_primary_group);
  const auto [ss_s, ss_e] = A.col_layout.group_offsets(_schur_group);
  for (std::size_t i = 0; i < pp_e - pp_s; ++i)
    x.tensors[pp_s + i] = x_p.tensors[i];
  for (std::size_t i = 0; i < ss_e - ss_s; ++i)
    x.tensors[ss_s + i] = x_s.tensors[i];

  return x;
}

SparseMatrix
SchurComplement::solve(const SparseMatrix & A, const SparseMatrix & B) const
{
  neml_assert_dbg(
      A.row_ngroup() >= 2, "SchurComplement requires at least 2 row groups, got ", A.row_ngroup());
  neml_assert_dbg(A.col_ngroup() >= 2,
                  "SchurComplement requires at least 2 column groups, got ",
                  A.col_ngroup());

  // Extract the four blocks using view layouts
  const auto A_pp = A.group(_primary_group, _primary_group);
  const auto A_ps = A.group(_primary_group, _schur_group);
  const auto A_sp = A.group(_schur_group, _primary_group);
  const auto A_ss = A.group(_schur_group, _schur_group);

  // Extract the two row groups of B by slicing tensors
  const auto [p_s, p_e] = A.row_layout.group_offsets(_primary_group);
  const auto [s_s, s_e] = A.row_layout.group_offsets(_schur_group);
  const SparseMatrix B_p(A.row_layout.group(_primary_group),
                         B.col_layout,
                         std::vector<std::vector<Tensor>>(B.tensors.begin() + Size(p_s),
                                                          B.tensors.begin() + Size(p_e)));
  const SparseMatrix B_s(A.row_layout.group(_schur_group),
                         B.col_layout,
                         std::vector<std::vector<Tensor>>(B.tensors.begin() + Size(s_s),
                                                          B.tensors.begin() + Size(s_e)));

  // Step 1: Y = A_pp^{-1} A_ps
  const auto Y = _primary_solver->solve(A_pp, A_ps);

  // Step 2: Z = A_pp^{-1} B_p
  const auto Z = _primary_solver->solve(A_pp, B_p);

  // Step 3: S = A_ss - A_sp Y
  const auto A_sp_f = A_sp.assemble(/*assemble_intmd=*/false);
  const auto Y_f = Y.assemble(/*assemble_intmd=*/false);
  const auto S_f = A_ss.assemble(/*assemble_intmd=*/false) - mm(A_sp_f, Y_f);

  // Step 4: D = B_s - A_sp Z
  const auto Z_f = Z.assemble(/*assemble_intmd=*/false);
  const auto D_f = B_s.assemble(/*assemble_intmd=*/false) - mm(A_sp_f, Z_f);

  // Wrap S and D for the schur_solver
  const auto schur_col = A.col_layout.group(_schur_group);
  SparseMatrix S_mat(schur_col, schur_col);
  S_mat.disassemble(S_f, /*assemble_intmd=*/false);
  SparseMatrix D_mat(schur_col, B.col_layout);
  D_mat.disassemble(D_f, /*assemble_intmd=*/false);

  // Step 5: X_s = S^{-1} D
  const auto X_s = _schur_solver->solve(S_mat, D_mat);

  // Step 6: X_p = Z - Y X_s
  const auto X_p_f = Z_f - mm(Y_f, X_s.assemble(/*assemble_intmd=*/false));
  const auto primary_col = A.col_layout.group(_primary_group);
  SparseMatrix X_p(primary_col, B.col_layout);
  X_p.disassemble(X_p_f, /*assemble_intmd=*/false);

  // Assemble the full solution
  SparseMatrix X(A.col_layout, B.col_layout);
  const auto [pp_s, pp_e] = A.col_layout.group_offsets(_primary_group);
  const auto [ss_s, ss_e] = A.col_layout.group_offsets(_schur_group);
  for (std::size_t i = 0; i < pp_e - pp_s; ++i)
    X.tensors[pp_s + i] = X_p.tensors[i];
  for (std::size_t i = 0; i < ss_e - ss_s; ++i)
    X.tensors[ss_s + i] = X_s.tensors[i];

  return X;
}

} // namespace neml2
