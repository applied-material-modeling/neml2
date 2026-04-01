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

#include "neml2/equation_systems/AssembledMatrix.h"
#include "neml2/equation_systems/AssembledVector.h"
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"
#include "neml2/tensors/functions/mm.h"
#include "neml2/tensors/functions/mv.h"
#include "neml2/tensors/functions/sum.h"

namespace neml2
{

AssembledMatrix::AssembledMatrix(AxisLayout rl, AxisLayout cl)
  : tensors(rl.ngroup(), std::vector<Tensor>(cl.ngroup())),
    row_layout(std::move(rl)),
    col_layout(std::move(cl))
{
}

AssembledMatrix::AssembledMatrix(AxisLayout rl, AxisLayout cl, std::vector<std::vector<Tensor>> ts)
  : tensors(std::move(ts)),
    row_layout(std::move(rl)),
    col_layout(std::move(cl))
{
  neml_assert_dbg(tensors.size() == row_layout.ngroup(),
                  "Number of matrix rows does not match row layout group size");
  for (std::size_t i = 0; i < row_layout.ngroup(); ++i)
    neml_assert_dbg(tensors[i].size() == col_layout.ngroup(),
                    "Number of matrix columns does not match column layout group size");
}

TensorOptions
AssembledMatrix::options() const
{
  for (const auto & row : tensors)
    for (const auto & t : row)
      if (t.defined())
        return t.options();
  return default_tensor_options();
}

AssembledMatrix
AssembledMatrix::group(std::size_t i, std::size_t j) const
{
  neml_assert(i < row_layout.ngroup(), "Row group index out of range");
  neml_assert(j < col_layout.ngroup(), "Column group index out of range");
  return AssembledMatrix(row_layout.group(i), col_layout.group(j), {{tensors[i][j]}});
}

SparseMatrix
AssembledMatrix::disassemble() const
{
  std::vector<std::vector<Tensor>> sp_tensors(row_layout.nvar(),
                                              std::vector<Tensor>(col_layout.nvar()));

  for (std::size_t grp_i = 0; grp_i < row_layout.ngroup(); ++grp_i)
    for (std::size_t grp_j = 0; grp_j < col_layout.ngroup(); ++grp_j)
    {
      const auto & t = tensors[grp_i][grp_j];
      const auto [istart, iend] = row_layout.group_offsets(grp_i);
      const auto [jstart, jend] = col_layout.group_offsets(grp_j);
      const auto istr = row_layout.istr(grp_i);
      const auto jstr = col_layout.istr(grp_j);
      const bool assemble_intmd =
          (istr == AxisLayout::IStructure::DENSE) && (jstr == AxisLayout::IStructure::DENSE);

      neml_assert_dbg(
          t.base_dim() == 2, "disassemble expects base dimension of 2, got ", t.base_dim());
      if (assemble_intmd)
        neml_assert_dbg(t.intmd_dim() == 0,
                        "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                        t.intmd_dim());

      const auto row_ss = row_layout.group(grp_i).storage_sizes(assemble_intmd);
      const auto col_ss = col_layout.group(grp_j).storage_sizes(assemble_intmd);
      const auto & D = t.dynamic_sizes();
      const auto I = t.intmd_dim();

      auto t_rows = t.split(row_ss, -2);
      for (std::size_t i = istart; i < iend; ++i)
      {
        auto t_cols = t_rows[i - istart].split(col_ss, -1);
        for (std::size_t j = jstart; j < jend; ++j)
        {
          auto tij = Tensor(t_cols[j - jstart], D, I);
          if (!assemble_intmd)
            sp_tensors[i][j] = tij.base_reshape(
                utils::add_shapes(row_layout.base_sizes(i), col_layout.base_sizes(j)));
          else
            sp_tensors[i][j] =
                from_assembly<2>(tij,
                                 {row_layout.intmd_sizes(i), col_layout.intmd_sizes(j)},
                                 {row_layout.base_sizes(i), col_layout.base_sizes(j)});
        }
      }
    }

  return SparseMatrix(row_layout, col_layout, std::move(sp_tensors));
}

AssembledMatrix
operator-(const AssembledMatrix & A)
{
  auto B = A;
  for (std::size_t i = 0; i < B.row_layout.ngroup(); ++i)
    for (std::size_t j = 0; j < B.col_layout.ngroup(); ++j)
      if (B.tensors[i][j].defined())
        B.tensors[i][j] = -B.tensors[i][j];
  return B;
}

AssembledMatrix
operator+(const AssembledMatrix & A, const AssembledMatrix & B)
{
  neml_assert_dbg(A.row_layout == B.row_layout, "Row layouts do not match");
  neml_assert_dbg(A.col_layout == B.col_layout, "Column layouts do not match");
  std::vector<std::vector<Tensor>> tensors(A.row_layout.ngroup(),
                                           std::vector<Tensor>(A.col_layout.ngroup()));
  for (std::size_t i = 0; i < A.row_layout.ngroup(); ++i)
    for (std::size_t j = 0; j < A.col_layout.ngroup(); ++j)
    {
      const auto & aij = A.tensors[i][j];
      const auto & bij = B.tensors[i][j];
      if (aij.defined() && bij.defined())
        tensors[i][j] = aij + bij;
      else if (aij.defined())
        tensors[i][j] = aij;
      else if (bij.defined())
        tensors[i][j] = bij;
    }
  return AssembledMatrix(A.row_layout, A.col_layout, std::move(tensors));
}

AssembledMatrix
operator-(const AssembledMatrix & A, const AssembledMatrix & B)
{
  neml_assert_dbg(A.row_layout == B.row_layout, "Row layouts do not match");
  neml_assert_dbg(A.col_layout == B.col_layout, "Column layouts do not match");
  std::vector<std::vector<Tensor>> tensors(A.row_layout.ngroup(),
                                           std::vector<Tensor>(A.col_layout.ngroup()));
  for (std::size_t i = 0; i < A.row_layout.ngroup(); ++i)
    for (std::size_t j = 0; j < A.col_layout.ngroup(); ++j)
    {
      const auto & aij = A.tensors[i][j];
      const auto & bij = B.tensors[i][j];
      if (aij.defined() && bij.defined())
        tensors[i][j] = aij - bij;
      else if (aij.defined())
        tensors[i][j] = aij;
      else if (bij.defined())
        tensors[i][j] = -bij;
    }
  return AssembledMatrix(A.row_layout, A.col_layout, std::move(tensors));
}

AssembledMatrix
operator*(const AssembledMatrix & A, const AssembledMatrix & B)
{
  neml_assert_dbg(A.col_layout == B.row_layout, "Mismatched layouts for matrix multiplication");
  std::vector<std::vector<Tensor>> tensors(A.row_layout.ngroup(),
                                           std::vector<Tensor>(B.col_layout.ngroup()));
  // Cij = sum_k Aik * Bkj
  for (std::size_t i = 0; i < A.row_layout.ngroup(); ++i)
    for (std::size_t j = 0; j < B.col_layout.ngroup(); ++j)
    {
      Tensor cij;
      for (std::size_t k = 0; k < A.col_layout.ngroup(); ++k)
      {
        const auto & aik = A.tensors[i][k];
        const auto & bkj = B.tensors[k][j];
        if (!aik.defined() || !bkj.defined())
          continue;
        auto r = neml2::mm(aik, bkj);

        // If the reduction along k is performed on IStructure::BLOCK, we also need to reduce along
        // the block intermediate dimensions
        const auto istr = A.col_layout.istr(k);
        if (istr == AxisLayout::IStructure::BLOCK)
        {
          neml_assert_dbg(
              r.intmd_dim() <= 2,
              "Expected at most 2 intermediate dimensions for BLOCK/BLOCK structure, got ",
              r.intmd_dim());
          if (r.intmd_dim() == 1)
            r = intmd_sum(r, {0}, /*keepdim=*/false);
          else if (r.intmd_dim() == 2)
            r = intmd_sum(r, {0, 1}, /*keepdim=*/false);
        }

        if (!cij.defined())
          cij = r;
        else
          cij = cij + r;
      }

      tensors[i][j] = cij;
    }
  return AssembledMatrix(A.row_layout, B.col_layout, std::move(tensors));
}

AssembledVector
operator*(const AssembledMatrix & A, const AssembledVector & b)
{
  neml_assert_dbg(A.col_layout == b.layout, "Mismatched layouts for matrix multiplication");
  std::vector<Tensor> tensors(A.row_layout.ngroup());
  // ci = sum_j Aij * bj
  for (std::size_t i = 0; i < A.row_layout.ngroup(); ++i)
  {
    Tensor cij;
    for (std::size_t j = 0; j < b.layout.ngroup(); ++j)
    {
      const auto & aij = A.tensors[i][j];
      const auto & bj = b.tensors[j];
      if (!aij.defined() || !bj.defined())
        continue;
      auto r = neml2::mv(aij, bj);

      // If the reduction along j is performed on IStructure::BLOCK, we also need to reduce along
      // the block intermediate dimensions
      const auto istr = A.col_layout.istr(j);
      if (istr == AxisLayout::IStructure::BLOCK)
      {
        neml_assert_dbg(
            r.intmd_dim() <= 2,
            "Expected at most 2 intermediate dimensions for BLOCK/BLOCK structure, got ",
            r.intmd_dim());
        if (r.intmd_dim() == 1)
          r = intmd_sum(r, {0}, /*keepdim=*/false);
        else if (r.intmd_dim() == 2)
          r = intmd_sum(r, {0, 1}, /*keepdim=*/false);
      }

      if (!cij.defined())
        cij = r;
      else
        cij = cij + r;
    }
    tensors[i] = cij;
  }
  return AssembledVector(A.row_layout, std::move(tensors));
}

} // namespace neml2
