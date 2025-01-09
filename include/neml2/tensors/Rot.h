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

namespace neml2
{
// Forward declarations
class Scalar;
class Vec;
class R2;
class R3;

/**
 * @brief Rotation stored as modified Rodrigues parameters
 *
 * The storage space is (3).
 *
 * One way to define this convention is that the three parameters are the values of the vector n
 * tan(theta/4) where n is the rotation axis and theta the rotation angle.
 *
 * Note this is different from the standard Rodrigues parameters, which are defined by n
 * tan(theta/2)
 */
class Rot : public VecBase<Rot>
{
public:
  using VecBase<Rot>::VecBase;

  Rot(const Vec & v);

  /// The identity rotation, helpfully the zero vector
  [[nodiscard]] static Rot identity(const TensorOptions & options = default_tensor_options());

  /// Fill from an array of Euler angles
  static Rot fill_euler_angles(const Vec & v,
                               const std::string & angle_convention,
                               const std::string & angle_type);

  /// Fill from rotation matrices
  static Rot fill_matrix(const R2 & M);

  /// Fill some number of random orientations
  static Rot fill_random(unsigned int n, Size random_seed);

  /// Fill from standard Rodrigues parameters
  static Rot fill_rodrigues(const Scalar & rx, const Scalar & ry, const Scalar & rz);

  /// Inversion
  Rot inverse() const;

  /// Generate a rotation matrix using the Euler-Rodrigues formula
  R2 euler_rodrigues() const;

  /// d(R2)/d(r) -- useful in constructing other derivatives
  R3 deuler_rodrigues() const;

  /// Rotate
  Rot rotate(const Rot & r) const;

  /// Derivative of the rotated Rodrigues vector w.r.t. the other Rodrigues vector
  R2 drotate(const Rot & r) const;

  /// Derivative of the rotated Rodrigues vector w.r.t. this vector
  R2 drotate_self(const Rot & r) const;

  /// Return the shadow parameter set (a set of MRPs that define the same orientation)
  Rot shadow() const;

  /// Return the derivative of the shadow map
  R2 dshadow() const;

  /// Distance measure between two rotations, accounting for shadow mapping
  Scalar dist(const Rot & r) const;

  /// Raw distance formula, not accounting for shadown mapping
  Scalar gdist(const Rot & r) const;

  /// Volume element at locations
  Scalar dV() const;
};

/// Composition of rotations r3 = r1 * r2 (r2 first, then r1)
//  So this follows the "matrix" convention where it's exactly the same
//  as the standard matrix product R1 * R2 where R1 and R2 are the
//  matrix representations of r1 and r2
Rot operator*(const Rot & r1, const Rot & r2);

} // namespace neml2
