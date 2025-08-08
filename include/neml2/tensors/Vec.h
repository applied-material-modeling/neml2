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

#include "neml2/tensors/VecBase.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/Transformable.h"

namespace neml2
{
class Scalar;
class R2;
class Rot;

/**
 * @brief 3-vector.
 *
 * The storage space is (3).
 */
class Vec : public VecBase<Vec>, public Transformable<Vec>
{
public:
  using VecBase<Vec>::VecBase;

  Vec(const Rot & r);

  /// The derivative of a Vec with respect to itself
  [[nodiscard]] static R2 identity_map(const TensorOptions & options = default_tensor_options());

  /// dot product
  Scalar dot(const Vec & v) const;

  /// cross product
  Vec cross(const Vec & v) const;

  /// outer product
  R2 outer(const Vec & v) const;

  /// self outer product
  SR2 self_outer() const;

  // Transform by a crystal symmetry operator
  Vec transform(const R2 & op) const override;
};
} // namespace neml2
