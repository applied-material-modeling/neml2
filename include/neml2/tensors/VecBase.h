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
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/functions/linalg/vector_norm.h"

namespace neml2
{
class Rot;
class R3;

/**
 * @brief Base class 3-vector.
 *
 * The storage space is (3). This class provides common operations for vector.
 */
template <class Derived>
class VecBase : public PrimitiveTensor<Derived, 3>
{
public:
  using PrimitiveTensor<Derived, 3>::PrimitiveTensor;

  [[nodiscard]] static Derived fill(const CScalar & v1,
                                    const CScalar & v2,
                                    const CScalar & v3,
                                    const TensorOptions & options = default_tensor_options());

  [[nodiscard]] static Derived fill(const Scalar & v1, const Scalar & v2, const Scalar & v3);

  /// The derivative of a vector with respect to itself
  [[nodiscard]] static R2 identity_map(const TensorOptions & options = default_tensor_options());

  /// Accessor
  Scalar operator()(Size i) const;

  /// Norm squared
  Scalar norm_sq() const;

  /// Norm
  Scalar norm() const;

  /// Rotate using a Rodrigues vector
  Derived rotate(const Rot & r) const;

  /// Rotate using a rotation matrix
  Derived rotate(const R2 & R) const;

  /// Derivative of the rotated vector w.r.t. the Rodrigues vector
  R2 drotate(const Rot & r) const;

  /// Derivative of the rotated vector w.r.t. the rotation matrix
  R3 drotate(const R2 & R) const;
};
} // namespace neml2
