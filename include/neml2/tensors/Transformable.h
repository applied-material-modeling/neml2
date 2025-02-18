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

#include "neml2/misc/types.h"
#include "neml2/misc/defaults.h"

namespace neml2
{
class R2;
class Quaternion;
class Rot;
class Vec;

/// @brief  Mixin class for things that can be transformed by a symmetry operator
/// @tparam Derived type
template <class Derived>
class Transformable
{
public:
  Transformable() = default;

  Transformable(const Transformable &) = default;
  Transformable(Transformable &&) noexcept = default;
  Transformable & operator=(const Transformable &) = default;
  Transformable & operator=(Transformable &&) noexcept = default;
  virtual ~Transformable() = default;

  /// @brief apply a transformation operator
  /// @param op the transformation operator
  /// @return an instance of the Derived type that has been transform
  virtual Derived transform(const R2 & op) const = 0;
};

/// Construct from quaternions, useful for comparison to old NEML
R2 transform_from_quaternion(const Quaternion & q);

/// The identity transformation, i.e.e the Rank2 identity tensor
R2 identity_transform(const TensorOptions & options = default_tensor_options());
/// A proper rotation, here provided by a Rot object
R2 proper_rotation_transform(const Rot & rot);
/// An improper rotation (rotation + reflection), here provided by a rot object giving the rotation and reflection axis
R2 improper_rotation_transform(const Rot & rot);
/// A reflection, defined by the reflection plane
R2 reflection_transform(const Vec & v);
/// An inversion center
R2 inversion_transform(const TensorOptions & options = default_tensor_options());
} // namespace neml2
