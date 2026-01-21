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

#include "neml2/equation_systems/HeterogeneousData.h"

namespace neml2
{
struct HMatrix : public HeterogeneousData
{
  HMatrix() = default;

  /// construct a zero HMatrix with given sub-block shapes
  HMatrix(const std::vector<TensorShapeRef> &, const std::vector<TensorShapeRef> &);
  HMatrix(std::vector<TensorShape>, std::vector<TensorShape>);

  /// construct a HMatrix from a matrix of sub-block Tensors and shapes of each sub-tensor
  HMatrix(const std::vector<std::vector<Tensor>> &,
          const std::vector<TensorShapeRef> &,
          const std::vector<TensorShapeRef> &);
  HMatrix(const std::vector<std::vector<Tensor>> &,
          std::vector<TensorShape>,
          std::vector<TensorShape>);

  /// Number of rows of sub-tensors
  std::size_t m() const { return _row_shapes.size(); }
  /// Number of columns of sub-tensors
  std::size_t n() const { return _col_shapes.size(); }
  /// Row shapes
  std::vector<TensorShapeRef> block_row_sizes() const;
  /// Column shapes
  std::vector<TensorShapeRef> block_col_sizes() const;
  /// Row shape of a sub-tensor
  TensorShapeRef block_row_sizes(std::size_t i) const { return _row_shapes[i]; }
  /// Column shape of a sub-tensor
  TensorShapeRef block_col_sizes(std::size_t j) const { return _col_shapes[j]; }
  ///@{
  /// Access to sub-tensors
  const Tensor & operator()(std::size_t i, std::size_t j) const;
  Tensor & operator()(std::size_t i, std::size_t j);
  ///@}

  /// Negation
  HMatrix operator-() const;

  /// Assemble into a dense, flat matrix
  std::tuple<Tensor, std::vector<Size>, std::vector<Size>>
  assemble(OptionalArrayRef<std::size_t> row_blocks = std::nullopt,
           OptionalArrayRef<std::size_t> col_blocks = std::nullopt) const;

  /// Disassemble a matrix into sub-blocks
  void disassemble(const Tensor &,
                   OptionalArrayRef<std::size_t> row_blocks = std::nullopt,
                   OptionalArrayRef<std::size_t> col_blocks = std::nullopt);

private:
  /// Construct from a flat vector of sub-tensors
  HMatrix(std::vector<Tensor> J,
          const std::vector<TensorShapeRef> & row_shapes,
          const std::vector<TensorShapeRef> & col_shapes);
  HMatrix(std::vector<Tensor> J,
          std::vector<TensorShape> row_shapes,
          std::vector<TensorShape> col_shapes);

  ///@{
  /// sub-block shapes
  std::vector<TensorShape> _row_shapes;
  std::vector<TensorShape> _col_shapes;
  ///@}
};
} // namespace neml2
