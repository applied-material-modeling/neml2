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
struct SparseMatrix;
struct AssembledVector;

/// Dense representation of a matrix assembled from a 2D-list of tensors and their layout
struct AssembledMatrix
{
  AssembledMatrix() = default;
  AssembledMatrix(AxisLayout, AxisLayout);
  AssembledMatrix(AxisLayout, AxisLayout, std::vector<std::vector<Tensor>>);

  /// Tensor options
  TensorOptions options() const;

  /// A block of the assembled matrix corresponding to row group i and column group j
  AssembledMatrix group(std::size_t, std::size_t) const;

  /// Disassemble into a 2D-list of tensors
  SparseMatrix disassemble() const;
  /// Assembled tensors for each block of the matrix
  std::vector<std::vector<Tensor>> tensors;
  /// Row layout of the tensors
  AxisLayout row_layout;
  /// Column layout of the tensors
  AxisLayout col_layout;
};

/// unary negation
AssembledMatrix operator-(const AssembledMatrix &);
/// binary addition
AssembledMatrix operator+(const AssembledMatrix &, const AssembledMatrix &);
/// binary subtraction
AssembledMatrix operator-(const AssembledMatrix &, const AssembledMatrix &);
/// Matrix-matrix product
AssembledMatrix operator*(const AssembledMatrix &, const AssembledMatrix &);
/// Matrix-vector product
AssembledVector operator*(const AssembledMatrix &, const AssembledVector &);

} // namespace neml2
