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

struct HVector : public HeterogeneousData
{
  /// construct a zero HVector with given sub-block shapes
  HVector(const std::vector<TensorShapeRef> &);
  HVector(std::vector<TensorShape>);

  /// construct a HVector from a vector of sub-block Tensors
  HVector(std::vector<Tensor>, const std::vector<TensorShapeRef> &);
  HVector(std::vector<Tensor>, std::vector<TensorShape>);

  /// construct a HVector by splitting an assembled vector into sub-blocks
  HVector(Tensor, const std::vector<TensorShapeRef> &);
  HVector(Tensor, std::vector<TensorShape>);

  /// Assign from disassembled data
  void operator=(const std::vector<Tensor> &);
  /// Assign from assembled data
  void operator=(Tensor);

  /// Number of sub-tensors
  std::size_t n() const { return _shapes.size(); }
  /// Sub-block shapes
  std::vector<TensorShapeRef> block_sizes() const;
  /// Shape of a sub-tensor
  TensorShapeRef block_sizes(std::size_t i) const;
  /// Split sizes of sub-blocks
  const std::vector<Size> & block_numel() const { return _numels; }
  /// Split size of a sub-block
  Size block_numel(std::size_t i) const;

  ///@{
  /// Access to sub-tensors
  const Tensor & operator[](std::size_t i) const;
  Tensor & operator[](std::size_t i);
  ///@}

  /// Negation
  HVector operator-() const;

  /// Update the data of the contained Tensors (without changing their computational graph)
  void update_data(const HVector & other);

  /// Update the contained Tensors
  void update(const HVector & other);

private:
  /// Assemble the existing sub-blocks into the assembled data cache
  void assemble() const override;

  /// Disassemble the existing assembled data into the sub-blocks cache
  void disassemble() const override;

  /// sub-block shapes
  std::vector<TensorShape> _shapes;

  /// sub-block numels
  std::vector<Size> _numels;
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
