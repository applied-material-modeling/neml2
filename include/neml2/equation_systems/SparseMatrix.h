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
#include "neml2/base/MutableArrayRef.h"

namespace neml2
{
struct SparseMatrixView;

/// Sparse representation of a matrix consisting of a 2D-list of tensors and their layout
struct SparseMatrix
{
  SparseMatrix(std::shared_ptr<AxisLayout>, std::shared_ptr<AxisLayout>);
  SparseMatrix(std::vector<Tensor>, std::shared_ptr<AxisLayout>, std::shared_ptr<AxisLayout>);

  /// Number of variables
  std::size_t size() const;
  /// Number of row variables
  std::size_t nrow() const;
  /// Number of column variables
  std::size_t ncol() const;
  /// Number of row variable groups
  std::size_t row_ngroup() const;
  /// Number of column variable groups
  std::size_t col_ngroup() const;
  /// Semi-contiguous view of a block of the sparse matrix
  SparseMatrixView group(std::size_t, std::size_t) const;
  /// List of tensors
  std::vector<Tensor> tensors;
  /// Row layout of the tensors, partitioned by variable groups
  std::shared_ptr<AxisLayout> row_layout;
  /// Column layout of the tensors, partitioned by variable groups
  std::shared_ptr<AxisLayout> col_layout;
};

struct SparseMatrixView
{
  /// Number of variables
  std::size_t size() const;
  /// Assemble into a Tensor with two base dimensions
  Tensor assemble(bool assemble_intmd) const;
  /// Disassemble a Tensor according to the axis layouts
  void disassemble(const Tensor &, bool assemble_intmd);
  /// View of the list of list of tensors
  std::vector<MutableArrayRef<Tensor>> tensors;
  /// View of the row layout of the tensors
  AxisLayoutView row_layout;
  /// View of the column layout of the tensors
  AxisLayoutView col_layout;
};

} // namespace neml2
