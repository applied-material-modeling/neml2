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

#include <cstdint>
#include "neml2/equation_systems/AxisLayout.h"
#include "neml2/tensors/Tensor.h"

namespace neml2
{
/// Sparse representation of a vector consisting of a list of tensors and their layout
struct SparseVector
{
  enum class IStructure : uint8_t;
  SparseVector() = default;
  SparseVector(AxisLayout, IStructure = IStructure::DENSE);
  SparseVector(AxisLayout, std::vector<Tensor>, IStructure = IStructure::DENSE);

  /// Tensor options
  TensorOptions options() const;

  /// Number of variable groups
  std::size_t ngroup() const;
  /// Contiguous view of the sparse vector
  SparseVector group(std::size_t) const;

  /// Number of variables
  std::size_t size() const;
  /// Assemble into a Tensor with one base dimension
  Tensor assemble() const;
  /// Disassemble a Tensor according to the axis layout
  void disassemble(const Tensor &);

  /// List of tensors
  std::vector<Tensor> tensors;
  /// Layout of the tensors
  AxisLayout layout;
  /// Structure represented by intermediate dimensions (if any)
  enum class IStructure : uint8_t
  {
    DENSE, ///< All intermediate dimensions are grouped into base dimensions
    BLOCK, ///< Intermediate dimensions represent blocks of variables
  } istr = IStructure::DENSE;
};

///@{
/// Unary negation
SparseVector operator-(const SparseVector &);
/// Binary addition
SparseVector operator+(const SparseVector &, const SparseVector &);
/// Binary subtraction
SparseVector operator-(const SparseVector &, const SparseVector &);
/// Multiplication with scalar
SparseVector operator*(const Scalar &, const SparseVector &);
SparseVector operator*(const SparseVector &, const Scalar &);
/// Inner product
Scalar operator*(const SparseVector &, const SparseVector &);
/// Norm-squared
Scalar norm_sq(const SparseVector &);
/// Norm
Scalar norm(const SparseVector &);
///@}
} // namespace neml2
