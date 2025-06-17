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

#include "neml2/tensors/PrimitiveTensor.h"

namespace neml2
{
class Scalar;
class Vec;
class SR2;
class R3;
class R4;
class Rot;
class WR2;
class R2;

template <class>
class VecBase;

/**
 * @brief Base class for second order tensor.
 *
 * The storage space is (3, 3).
 */
template <class Derived>
class R2Base : public PrimitiveTensor<Derived, 3, 3>
{
public:
  using PrimitiveTensor<Derived, 3, 3>::PrimitiveTensor;

  /// Conversion operator to symmetric second order tensor
  explicit operator SR2() const;

  /// Fill the diagonals with a11 = a22 = a33 = a
  [[nodiscard]] static Derived fill(const CScalar & a,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived fill(const Scalar & a);
  /// Fill the diagonals with a11, a22, a33
  [[nodiscard]] static Derived fill(const CScalar & a11,
                                    const CScalar & a22,
                                    const CScalar & a33,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived fill(const Scalar & a11, const Scalar & a22, const Scalar & a33);
  /// Fill symmetric entries
  [[nodiscard]] static Derived fill(const CScalar & a11,
                                    const CScalar & a22,
                                    const CScalar & a33,
                                    const CScalar & a23,
                                    const CScalar & a13,
                                    const CScalar & a12,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived fill(const Scalar & a11,
                                    const Scalar & a22,
                                    const Scalar & a33,
                                    const Scalar & a23,
                                    const Scalar & a13,
                                    const Scalar & a12);
  /// Fill all entries
  [[nodiscard]] static Derived fill(const CScalar & a11,
                                    const CScalar & a12,
                                    const CScalar & a13,
                                    const CScalar & a21,
                                    const CScalar & a22,
                                    const CScalar & a23,
                                    const CScalar & a31,
                                    const CScalar & a32,
                                    const CScalar & a33,
                                    const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static Derived fill(const Scalar & a11,
                                    const Scalar & a12,
                                    const Scalar & a13,
                                    const Scalar & a21,
                                    const Scalar & a22,
                                    const Scalar & a23,
                                    const Scalar & a31,
                                    const Scalar & a32,
                                    const Scalar & a33);
  /// Skew matrix from Vec
  [[nodiscard]] static Derived skew(const Vec & v);
  /// Identity
  [[nodiscard]] static Derived identity(const TensorOptions & options = default_tensor_options());

  /// Rotate using a Rodrigues vector
  Derived rotate(const Rot & r) const;

  /// Rotate using a rotation matrix
  Derived rotate(const R2 & R) const;

  /// Derivative of the rotated tensor w.r.t. the Rodrigues vector
  R3 drotate(const Rot & r) const;

  /// Derivative of the rotated tensor w.r.t. the rotation matrix
  R4 drotate(const R2 & R) const;

  /// Accessor
  Scalar operator()(Size i, Size j) const;

  /// Determinant
  Scalar det() const;

  /// Double contraction ij,ij
  Scalar inner(const R2 & other) const;

  /// Inversion
  Derived inverse() const;

  /// transpose
  Derived transpose() const;
};

/// matrix-vector product
// TODO: Fix the return type
template <class Derived1,
          class Derived2,
          typename = typename std::enable_if_t<std::is_base_of_v<R2Base<Derived1>, Derived1>>,
          typename = typename std::enable_if_t<std::is_base_of_v<VecBase<Derived2>, Derived2>>>
Vec operator*(const Derived1 & A, const Derived2 & b);

/// @brief matrix-matrix product
// TODO: Fix the return type
template <class Derived1,
          class Derived2,
          typename = typename std::enable_if_t<std::is_base_of_v<R2Base<Derived1>, Derived1>>,
          typename = typename std::enable_if_t<std::is_base_of_v<R2Base<Derived2>, Derived2>>>
R2 operator*(const Derived1 & A, const Derived2 & B);

} // namespace neml2
