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

#pragma once

#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
struct SparseVector;

/// Sparse representation of a matrix consisting of a 2D-list of tensors and their layout
struct SparseMatrix
{
  enum class IStructure : std::uint8_t;
  SparseMatrix() = default;
  SparseMatrix(AxisLayout, AxisLayout, IStructure = IStructure::DENSE);
  SparseMatrix(AxisLayout,
               AxisLayout,
               std::vector<std::vector<Tensor>>,
               IStructure = IStructure::DENSE);

  /// Tensor options
  TensorOptions options() const;

  /// Number of row variable groups
  std::size_t row_ngroup() const;
  /// Number of column variable groups
  std::size_t col_ngroup() const;
  /// Rows belonging to row group i, all columns
  SparseMatrix row_group(std::size_t i) const;
  /// All rows, columns belonging to column group j
  SparseMatrix col_group(std::size_t j) const;
  /// Semi-contiguous view of a block of the sparse matrix
  SparseMatrix group(std::size_t, std::size_t) const;

  /// Number of variables
  std::size_t size() const;
  /// Number of row variables
  std::size_t nrow() const;
  /// Number of column variables
  std::size_t ncol() const;
  /// Assemble into a Tensor with two base dimensions
  Tensor assemble() const;
  /// Disassemble a Tensor according to the axis layouts
  void disassemble(const Tensor &);
  /// 2D-list of tensors
  std::vector<std::vector<Tensor>> tensors;
  /// Row layout of the tensors, partitioned by variable groups
  AxisLayout row_layout;
  /// Column layout of the tensors, partitioned by variable groups
  AxisLayout col_layout;
  /// Structure represented by intermediate dimensions (if any)
  enum class IStructure : std::uint8_t
  {
    DENSE,          ///< All intermediate dimensions are grouped into base dimensions
    BLOCK_DIAGONAL, ///< Intermediate dimensions represent diagonal blocks of variables
  } istr = IStructure::DENSE;
};

/// Unary negation
SparseMatrix operator-(const SparseMatrix &);
/// Binary subtraction
SparseMatrix operator-(const SparseMatrix &, const SparseMatrix &);
/// Matrix-matrix product
SparseMatrix operator*(const SparseMatrix &, const SparseMatrix &);
/// Matrix-vector product
SparseVector operator*(const SparseMatrix &, const SparseVector &);

} // namespace neml2
