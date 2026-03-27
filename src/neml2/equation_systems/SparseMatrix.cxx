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
#include "neml2/equation_systems/assembly.h"
#include "neml2/misc/assertions.h"
#include "neml2/misc/defaults.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2
{

SparseMatrix::SparseMatrix(const AxisLayout & rl, const AxisLayout & cl)
  : tensors(rl.size() * cl.size()),
    row_layout(rl),
    col_layout(cl)
{
}

SparseMatrix::SparseMatrix(const AxisLayout & rl,
                           const AxisLayout & cl,
                           std::vector<std::vector<Tensor>> ts)
  : tensors(std::move(ts)),
    row_layout(rl),
    col_layout(cl)
{
  neml_assert_dbg(tensors.size() == row_layout.size(),
                  "Number of matrix rows does not match row layout size");
  for (std::size_t i = 0; i < row_layout.size(); ++i)
    neml_assert_dbg(tensors[i].size() == col_layout.size(),
                    "Number of matrix columns does not match column layout size");
}

std::size_t
SparseMatrix::row_ngroup() const
{
  return row_layout.ngroup();
}

std::size_t
SparseMatrix::col_ngroup() const
{
  return col_layout.ngroup();
}

SparseMatrix
SparseMatrix::group(std::size_t i, std::size_t j) const
{
  neml_assert_dbg(i < row_ngroup(), "Row group index out of range");
  neml_assert_dbg(j < col_ngroup(), "Column group index out of range");

  auto [row_start, row_end] = row_layout.group_offsets(i);
  auto [col_start, col_end] = col_layout.group_offsets(j);

  std::vector<std::vector<Tensor>> rows;
  rows.reserve(row_end - row_start);

  for (auto i = row_start; i < row_end; ++i)
  {
    std::vector<Tensor> row(tensors[i].begin() + Size(col_start),
                            tensors[i].begin() + Size(col_end));
    rows.push_back(std::move(row));
  }

  return SparseMatrix(row_layout.group(i), col_layout.group(j), std::move(rows));
}

std::size_t
SparseMatrix::size() const
{
  return tensors.size();
}

std::size_t
SparseMatrix::nrow() const
{
  return row_layout.size();
}

std::size_t
SparseMatrix::ncol() const
{
  return col_layout.size();
}

Tensor
SparseMatrix::assemble(bool assemble_intmd) const
{
  const auto row_ss = row_layout.storage_sizes(assemble_intmd);
  const auto col_ss = col_layout.storage_sizes(assemble_intmd);
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
      const auto & tij = tensors[i][j];
      if (!tij.defined())
        continue;

      if (!found_opt)
      {
        opt = tij.options();
        found_opt = true;
      }

      if (!assemble_intmd)
        tf_cols[j] = tij.base_reshape(
            {utils::numel(row_layout.base_sizes(i)), utils::numel(col_layout.base_sizes(j))});
      else
        tf_cols[j] = to_assembly<2>(tij,
                                    {row_layout.intmd_sizes(i), col_layout.intmd_sizes(j)},
                                    {row_layout.base_sizes(i), col_layout.base_sizes(j)});
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

  return base_cat(tf_rows, -2);
}

void
SparseMatrix::disassemble(const Tensor & t, bool assemble_intmd)
{
  neml_assert_dbg(t.base_dim() == 2, "disassemble expects base dimension of 2, got ", t.base_dim());
  if (assemble_intmd)
    neml_assert_dbg(t.intmd_dim() == 0,
                    "disassemble with intermediate shapes expects intmd dimension of 0, got ",
                    t.intmd_dim());

  const auto row_ss = row_layout.storage_sizes(assemble_intmd);
  const auto col_ss = col_layout.storage_sizes(assemble_intmd);
  const auto & D = t.dynamic_sizes();
  const auto I = t.intmd_dim();
  const auto m = row_ss.size();
  const auto n = col_ss.size();

  auto t_rows = t.split(row_ss, -2);
  for (std::size_t i = 0; i < m; ++i)
  {
    auto t_cols = t_rows[i].split(col_ss, -1);
    for (std::size_t j = 0; j < n; ++j)
    {
      auto tij = Tensor(t_cols[j], D, I);
      if (!assemble_intmd)
        tensors[i][j] =
            tij.base_reshape(utils::add_shapes(row_layout.base_sizes(i), col_layout.base_sizes(j)));
      else
        tensors[i][j] = from_assembly<2>(tij,
                                         {row_layout.intmd_sizes(i), col_layout.intmd_sizes(j)},
                                         {row_layout.base_sizes(i), col_layout.base_sizes(j)});
    }
  }
}

} // namespace neml2
