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

#include "neml2/equation_systems/SparseMatrix.h"
#include "neml2/equation_systems/AssembledMatrix.h"
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{

SparseMatrix::SparseMatrix(AxisLayout rl, AxisLayout cl)
  : tensors(rl.nvar(), std::vector<Tensor>(cl.nvar())),
    row_layout(std::move(rl)),
    col_layout(std::move(cl))
{
}

SparseMatrix::SparseMatrix(AxisLayout rl, AxisLayout cl, std::vector<std::vector<Tensor>> ts)
  : tensors(std::move(ts)),
    row_layout(std::move(rl)),
    col_layout(std::move(cl))
{
  neml_assert_dbg(tensors.size() == row_layout.nvar(),
                  "Number of matrix rows does not match row layout size");
  for (std::size_t i = 0; i < row_layout.nvar(); ++i)
    neml_assert_dbg(tensors[i].size() == col_layout.nvar(),
                    "Number of matrix columns does not match column layout size");
}

TensorOptions
SparseMatrix::options() const
{
  for (const auto & row : tensors)
    for (const auto & t : row)
      if (t.defined())
        return t.options();
  return default_tensor_options();
}

SparseMatrix
SparseMatrix::group(std::size_t i, std::size_t j) const
{
  neml_assert(i < row_layout.ngroup(), "Row group index out of range");
  neml_assert(j < col_layout.ngroup(), "Column group index out of range");
  auto [row_start, row_end] = row_layout.group_offsets(i);
  auto [col_start, col_end] = col_layout.group_offsets(j);
  std::vector<std::vector<Tensor>> rows;
  rows.reserve(row_end - row_start);
  for (auto r = row_start; r < row_end; ++r)
  {
    std::vector<Tensor> row(tensors[r].begin() + Size(col_start),
                            tensors[r].begin() + Size(col_end));
    rows.push_back(std::move(row));
  }
  return SparseMatrix(row_layout.group(i), col_layout.group(j), std::move(rows));
}

AssembledMatrix
SparseMatrix::assemble() const
{
  std::vector<std::vector<Tensor>> asm_tensors(row_layout.ngroup(),
                                               std::vector<Tensor>(col_layout.ngroup()));

  for (std::size_t grp_i = 0; grp_i < row_layout.ngroup(); ++grp_i)
    for (std::size_t grp_j = 0; grp_j < col_layout.ngroup(); ++grp_j)
    {
      const auto row_glayout = row_layout.group(grp_i);
      const auto col_glayout = col_layout.group(grp_j);
      const auto [istart, iend] = row_layout.group_offsets(grp_i);
      const auto [jstart, jend] = col_layout.group_offsets(grp_j);
      const auto istr = row_layout.group_istr(grp_i);
      const auto jstr = col_layout.group_istr(grp_j);
      neml_assert(
          istr == jstr,
          "Current implementation requires matching structure types for row and column groups");
      const bool assemble_intmd = (istr == AxisLayout::IStructure::DENSE);

      const auto row_ss = row_glayout.storage_sizes(assemble_intmd);
      const auto col_ss = col_glayout.storage_sizes(assemble_intmd);
      const auto m = row_ss.size();
      const auto n = col_ss.size();

      auto opt = default_tensor_options();
      bool found_opt = false;

      std::vector<Tensor> tf_rows(m);
      for (std::size_t i = 0; i < m; ++i)
      {
        std::vector<Tensor> tf_cols(n);
        for (std::size_t j = 0; j < n; ++j)
        {
          const auto & tij = tensors[istart + i][jstart + j];
          if (!tij.defined())
            continue;

          if (!found_opt)
          {
            opt = tij.options();
            found_opt = true;
          }

          if (!assemble_intmd)
            tf_cols[j] = tij.base_reshape(
                {utils::numel(row_glayout.base_sizes(i)), utils::numel(col_glayout.base_sizes(j))});
          else
            tf_cols[j] = to_assembly<2>(tij,
                                        {row_glayout.intmd_sizes(i), col_glayout.intmd_sizes(j)},
                                        {row_glayout.base_sizes(i), col_glayout.base_sizes(j)});
        }

        const auto dynamic_sizes = utils::broadcast_dynamic_sizes(tf_cols);
        const auto intmd_sizes = utils::broadcast_intmd_sizes(tf_cols);
        for (std::size_t j = 0; j < n; ++j)
        {
          auto & tfij = tf_cols[j];
          if (tfij.defined())
            tfij = tfij.batch_expand(dynamic_sizes, intmd_sizes);
          else
            tfij = Tensor::zeros(dynamic_sizes, intmd_sizes, {row_ss[i], col_ss[j]}, opt);
        }

        tf_rows[i] = base_cat(tf_cols, -1);
      }

      const auto col_ndof = std::accumulate(col_ss.begin(), col_ss.end(), Size(0), std::plus<>());
      const auto dynamic_sizes = utils::broadcast_dynamic_sizes(tf_rows);
      const auto intmd_sizes = utils::broadcast_intmd_sizes(tf_rows);
      for (std::size_t i = 0; i < m; ++i)
      {
        auto & tfi = tf_rows[i];
        if (tfi.defined())
          tfi = tfi.batch_expand(dynamic_sizes, intmd_sizes);
        else
          tfi = Tensor::zeros(dynamic_sizes, intmd_sizes, {row_ss[i], col_ndof}, opt);
      }

      asm_tensors[grp_i][grp_j] = base_cat(tf_rows, -2);
    }

  return AssembledMatrix(row_layout, col_layout, std::move(asm_tensors));
}

} // namespace neml2
