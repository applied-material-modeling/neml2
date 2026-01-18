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
class R2;
class SFR3;
class SSR4;
class SFFR4;
class Rot;
class SWR4;
class WR2;

/**
 * @brief The symmetric second order tensor.
 *
 * Mandel notation is used, and so the storage space is (6).
 */
class SR2 : public PrimitiveTensor<SR2, 6>
{
public:
  using PrimitiveTensor<SR2, 6>::PrimitiveTensor;

  /// Symmetrize an R2 then fill
  SR2(const R2 & T);

  /// Fill the diagonals with a11 = a22 = a33 = a
  [[nodiscard]] static SR2 fill(const CScalar & a,
                                const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static SR2 fill(const Scalar & a);
  /// Fill the diagonals with a11, a22, a33
  [[nodiscard]] static SR2 fill(const CScalar & a11,
                                const CScalar & a22,
                                const CScalar & a33,
                                const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static SR2 fill(const Scalar & a11, const Scalar & a22, const Scalar & a33);
  /// Fill all entries
  [[nodiscard]] static SR2 fill(const CScalar & a11,
                                const CScalar & a22,
                                const CScalar & a33,
                                const CScalar & a23,
                                const CScalar & a13,
                                const CScalar & a12,
                                const TensorOptions & options = default_tensor_options());
  [[nodiscard]] static SR2 fill(const Scalar & a11,
                                const Scalar & a22,
                                const Scalar & a33,
                                const Scalar & a23,
                                const Scalar & a13,
                                const Scalar & a12);
  /// Identity
  [[nodiscard]] static SR2 identity(const TensorOptions & options = default_tensor_options());

  /// Rotate
  SR2 rotate(const Rot & r) const;
  SR2 rotate(const R2 & R) const;

  /// Derivative of the rotated tensor w.r.t. the Rodrigues vector
  SFR3 drotate(const Rot & r) const;
  SFFR4 drotate(const R2 & R) const;

  /// Accessor
  using PrimitiveTensor<SR2, 6>::operator();
  Scalar operator()(Size i, Size j) const;

  /// Transpose, no-op
  SR2 transpose() const;
};

} // namespace neml2
