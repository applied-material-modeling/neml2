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

#include "neml2/tensors/equation_system/Matrix.h"
#include "neml2/misc/assertions.h"
#include "neml2/tensors/functions/cat.h"
#include "neml2/tensors/shape_utils.h"

namespace neml2::es
{

static std::vector<Tensor>
flatten_2D_vector(const std::vector<std::vector<Tensor>> & J, std::size_t m, std::size_t n)
{
  neml_assert_dbg(J.size() == m, "es::Matrix: data nrow = ", J.size(), ", expected ", m);
  std::vector<Tensor> v(m * n);
  std::size_t i = 0;
  for (const auto & row : J)
  {
    neml_assert(row.size() == n, "es::Matrix: data ncol = ", row.size(), ", expected ", n);
    for (const auto & Jij : row)
      v[i++] = Jij;
  }
  return v;
}

Matrix::Matrix(const std::vector<TensorShapeRef> & row_shapes,
               const std::vector<TensorShapeRef> & col_shapes)
  : _row_shapes(shape_refs_to_shapes(row_shapes)),
    _col_shapes(shape_refs_to_shapes(col_shapes))
{
  _data.resize(m() * n());
}

Matrix::Matrix(std::vector<TensorShape> row_shapes, std::vector<TensorShape> col_shapes)
  : _row_shapes(std::move(row_shapes)),
    _col_shapes(std::move(col_shapes))
{
  _data.resize(m() * n());
}

Matrix::Matrix(const std::vector<std::vector<Tensor>> & J,
               const std::vector<TensorShapeRef> & row_shapes,
               const std::vector<TensorShapeRef> & col_shapes)
  : ESData(flatten_2D_vector(J, row_shapes.size(), col_shapes.size())),
    _row_shapes(shape_refs_to_shapes(row_shapes)),
    _col_shapes(shape_refs_to_shapes(col_shapes))
{
}

Matrix::Matrix(const std::vector<std::vector<Tensor>> & J,
               std::vector<TensorShape> row_shapes,
               std::vector<TensorShape> col_shapes)
  : ESData(flatten_2D_vector(J, row_shapes.size(), col_shapes.size())),
    _row_shapes(std::move(row_shapes)),
    _col_shapes(std::move(col_shapes))
{
}

Matrix::Matrix(std::vector<Tensor> J,
               const std::vector<TensorShapeRef> & row_shapes,
               const std::vector<TensorShapeRef> & col_shapes)
  : ESData(std::move(J)),
    _row_shapes(shape_refs_to_shapes(row_shapes)),
    _col_shapes(shape_refs_to_shapes(col_shapes))
{
}

Matrix::Matrix(std::vector<Tensor> J,
               std::vector<TensorShape> row_shapes,
               std::vector<TensorShape> col_shapes)
  : ESData(std::move(J)),
    _row_shapes(std::move(row_shapes)),
    _col_shapes(std::move(col_shapes))
{
}

std::vector<TensorShapeRef>
Matrix::block_row_sizes() const
{
  return shapes_to_shape_refs(_row_shapes);
}

std::vector<TensorShapeRef>
Matrix::block_col_sizes() const
{
  return shapes_to_shape_refs(_col_shapes);
}

const Tensor &
Matrix::operator()(std::size_t i, std::size_t j) const
{
  neml_assert(i < m(), "es::Matrix::operator(): row index (", i, ") out of bounds (", m(), ")");
  neml_assert(j < n(), "es::Matrix::operator(): column index (", j, ") out of bounds (", n(), ")");
  return _data[i * n() + j];
}

Tensor &
Matrix::operator()(std::size_t i, std::size_t j)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<Tensor &>(std::as_const(*this).operator()(i, j));
}

Matrix
Matrix::operator-() const
{
  std::vector<Tensor> v(m() * n());
  for (size_t i = 0; i < m() * n(); i++)
  {
    const auto & vi = _data[i];
    if (vi.defined())
      v[i] = -vi;
  }
  return Matrix(v, block_row_sizes(), block_col_sizes());
}

std::tuple<Tensor, std::vector<Size>, std::vector<Size>>
Matrix::assemble(OptionalArrayRef<std::size_t> row_blocks,
                 OptionalArrayRef<std::size_t> col_blocks) const
{
  const auto row_indices = select_subblock_indices(row_blocks, m());
  const auto col_indices = select_subblock_indices(col_blocks, n());
  const auto row_shapes = select_shapes(_row_shapes, row_indices);
  const auto col_shapes = select_shapes(_col_shapes, col_indices);
  const auto row_split_sizes = numel(row_shapes);
  const auto col_split_sizes = numel(col_shapes);

  // If a variable is not found, tensor at that position remains undefined, and all undefined tensor
  // will later be filled with zeros.
  std::vector<Tensor> vs_f_rows(row_indices.size());
  for (std::size_t i = 0; i < row_indices.size(); ++i)
  {
    std::vector<Tensor> vs_f_cols(col_indices.size());
    for (std::size_t j = 0; j < col_indices.size(); ++j)
    {
      const auto & v = (*this)(row_indices[i], col_indices[j]);
      if (!v.defined())
        continue;
      neml_assert_dbg(
          v.intmd_dim() == 0,
          "cannot assemble es::Matrix with a sub-block having intermediate dimension(s)");
      const auto v_f = v.base_reshape({row_split_sizes[i], col_split_sizes[j]});
      vs_f_cols[j] = v_f;
    }

    // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with
    // zeros.
    const auto dynamic_sizes = utils::broadcast_dynamic_sizes(vs_f_cols);
    for (std::size_t j = 0; j < col_indices.size(); ++j)
    {
      auto & v = vs_f_cols[j];
      if (v.defined())
        v = v.dynamic_expand(dynamic_sizes);
      else
        v = Tensor::zeros(dynamic_sizes, {}, {row_split_sizes[i], col_split_sizes[j]}, options());
    }

    vs_f_rows[i] = base_cat(vs_f_cols, -1);
  }

  // Expand defined tensors with the broadcast dynamic shape and fill undefined tensors with zeros.
  const auto dynamic_sizes = utils::broadcast_dynamic_sizes(vs_f_rows);
  for (std::size_t i = 0; i < row_indices.size(); ++i)
  {
    auto & v = vs_f_rows[i];
    if (v.defined())
      v = v.dynamic_expand(dynamic_sizes);
    else
      v = Tensor::zeros(dynamic_sizes, {}, {row_split_sizes[i], col_split_sizes[i]}, options());
  }

  return {base_cat(vs_f_rows, -2), row_split_sizes, col_split_sizes};
}

void
Matrix::disassemble(const Tensor & mat,
                    OptionalArrayRef<std::size_t> row_blocks,
                    OptionalArrayRef<std::size_t> col_blocks)
{
  const auto row_indices = select_subblock_indices(row_blocks, m());
  const auto col_indices = select_subblock_indices(col_blocks, n());
  const auto row_shapes = select_shapes(_row_shapes, row_indices);
  const auto col_shapes = select_shapes(_col_shapes, col_indices);
  const auto row_split_sizes = numel(row_shapes);
  const auto col_split_sizes = numel(col_shapes);
  const auto & S = mat.dynamic_sizes();

  // split into rows
  auto mat_rows = mat.split(row_split_sizes, -2);
  for (std::size_t i = 0; i < row_indices.size(); ++i)
  {
    // split into columns
    auto mat_cols = mat_rows[i].split(col_split_sizes, -1);
    for (std::size_t j = 0; j < col_indices.size(); ++j)
      _data[row_indices[i] * n() + col_indices[j]] =
          Tensor(mat_cols[j], S).base_reshape(utils::add_shapes(row_shapes[i], col_shapes[j]));
  }
}

} // namespace neml2::es
