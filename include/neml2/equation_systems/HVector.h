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
std::vector<TensorShape> shape_refs_to_shapes(const std::vector<TensorShapeRef> & shape_refs);
std::vector<TensorShapeRef> shapes_to_shape_refs(const std::vector<TensorShape> & shapes);

std::vector<std::size_t> select_subblock_indices(OptionalArrayRef<std::size_t> blocks,
                                                 std::size_t n);
std::vector<TensorShapeRef> select_shapes(const std::vector<TensorShape> & all_shapes,
                                          ArrayRef<std::size_t> blocks);
std::vector<Size> numel(const std::vector<TensorShapeRef> & shapes);

struct HVector : public HeterogeneousData
{
  HVector() = default;

  /// construct a zero HVector with given sub-block shapes
  HVector(const std::vector<TensorShapeRef> &);
  HVector(std::vector<TensorShape>);

  /// construct a HVector from a vector of sub-block Tensors
  HVector(std::vector<Tensor>, const std::vector<TensorShapeRef> &);
  HVector(std::vector<Tensor>, std::vector<TensorShape>);

  /// Number of sub-tensors
  std::size_t n() const { return _shapes.size(); }
  /// Sub-block shapes
  std::vector<TensorShapeRef> block_sizes() const;
  /// Shape of a sub-tensor
  TensorShapeRef block_sizes(std::size_t i) const { return _shapes[i]; }

  ///@{
  /// Access to sub-tensors
  const Tensor & operator[](std::size_t i) const;
  Tensor & operator[](std::size_t i);
  ///@}

  /// Negation
  HVector operator-() const;

  /// Assemble into a dense, flat vector
  std::pair<Tensor, std::vector<Size>>
  assemble(OptionalArrayRef<std::size_t> blocks = std::nullopt) const;

  /// Take an assembled vector and split it into sub-blocks
  void disassemble(const Tensor &, OptionalArrayRef<std::size_t> blocks = std::nullopt);

  /// Update the data of the contained Tensors (without changing their computational graph)
  void update_data(const HVector & other);

  /// Update the contained Tensors
  void update(const HVector & other);

private:
  /// sub-block shapes
  std::vector<TensorShape> _shapes;
};

///@{
/// operators
HVector operator+(const HVector & a, const HVector & b);
HVector operator+(const Scalar & a, const HVector & b);
HVector operator+(const HVector & a, const Scalar & b);
HVector operator+(const CScalar & a, const HVector & b);
HVector operator+(const HVector & a, const CScalar & b);

HVector operator-(const HVector & a, const HVector & b);
HVector operator-(const Scalar & a, const HVector & b);
HVector operator-(const HVector & a, const Scalar & b);
HVector operator-(const CScalar & a, const HVector & b);
HVector operator-(const HVector & a, const CScalar & b);

Scalar operator*(const HVector & A, const HVector & b);
HVector operator*(const Scalar & a, const HVector & b);
HVector operator*(const HVector & a, const Scalar & b);
HVector operator*(const CScalar & a, const HVector & b);
HVector operator*(const HVector & a, const CScalar & b);

HVector operator/(const HVector & a, const Scalar & b);
HVector operator/(const HVector & a, const CScalar & b);
///@}

/// squared vector-norm of an HVector
Scalar norm_sq(const HVector & v);

/// vector-norm of an HVector
Scalar norm(const HVector & v);

} // namespace neml2
