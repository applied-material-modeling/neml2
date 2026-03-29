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
#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/shape_utils.h"

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
      const auto istr = row_layout.group_istr(grp_i);
      const auto jstr = col_layout.group_istr(grp_j);
      neml_assert(
          istr == jstr,
          "Current implementation requires matching structure types for row and column groups");
      const bool assemble_intmd = (istr == AxisLayout::IStructure::DENSE);

      neml_assert_dbg(
          t.base_dim() == 2, "disassemble expects base dimension of 2, got ", t.base_dim());
      if (assemble_intmd)
        neml_assert_dbg(t.intmd_dim() == 0,
                        "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                        t.intmd_dim());

      const auto row_ss = row_layout.storage_sizes(assemble_intmd);
      const auto col_ss = col_layout.storage_sizes(assemble_intmd);
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

} // namespace neml2
