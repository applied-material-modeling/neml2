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

#include <ATen/ops/linalg_vecdot.h>
#include <ATen/ops/linalg_cross.h>

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

  [[nodiscard]] static Derived
  fill(const Real & v1,
       const Real & v2,
       const Real & v3,
       const torch::TensorOptions & options = default_tensor_options());

  [[nodiscard]] static Derived fill(const Scalar & v1, const Scalar & v2, const Scalar & v3);

  /// The derivative of a vector with respect to itself
  [[nodiscard]] static R2
  identity_map(const torch::TensorOptions & options = default_tensor_options());

  /// Accessor
  Scalar operator()(Size i) const;

  /// dot product
  template <class Derived2>
  Scalar dot(const VecBase<Derived2> & v) const;

  /// cross product
  template <class Derived2>
  Derived cross(const VecBase<Derived2> & v) const;

  /// outer product
  template <class Derived2>
  R2 outer(const VecBase<Derived2> & v) const;

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

template <class Derived>
template <class Derived2>
Scalar
VecBase<Derived>::dot(const VecBase<Derived2> & v) const
{
  neml_assert_broadcastable_dbg(*this, v);
  auto res = torch::linalg_vecdot(*this, v);
  return Scalar(res, res.dim());
}

template <class Derived>
template <class Derived2>
Derived
VecBase<Derived>::cross(const VecBase<Derived2> & v) const
{
  neml_assert_broadcastable_dbg(*this, v);

  auto batch_dim = broadcast_batch_dim(*this, v);
  auto pair = torch::broadcast_tensors({*this, v});

  return Derived(torch::linalg_cross(pair[0], pair[1]), batch_dim);
}

template <class Derived>
template <class Derived2>
R2
VecBase<Derived>::outer(const VecBase<Derived2> & v) const
{
  return torch::matmul(this->unsqueeze(-1), v.unsqueeze(-2));
}
} // namespace neml2
