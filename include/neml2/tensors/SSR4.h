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
class SR2;
class R4;
class SSFR5;
class Rot;
class SSSSR8;

/**
 * @brief The symmetric fourth order tensor, with symmetry in the first two dimensionss as
 * well as in the last two dimensions.
 *
 * Mandel notation is used, and so the storage space is (6, 6).
 */
class SSR4 : public PrimitiveTensor<SSR4, 6, 6>
{
public:
  using PrimitiveTensor<SSR4, 6, 6>::PrimitiveTensor;

  /// Initialize with the symmetrized fourth order tensor
  SSR4(const R4 & T);

  /// Create the identity tensor \f$\delta_{ij}\delta_{kl}\f$
  [[nodiscard]] static SSR4 identity(const TensorOptions & options = default_tensor_options());
  /// Create the symmetric identity tensor \f$\delta_{ik}\delta_{jl}/2 + \delta_{il}\delta_{jk}/2\f$
  static SSR4 identity_sym(const TensorOptions & options = default_tensor_options());
  /// Create the volumetric identity tensor \f$\delta_{ij}\delta_{kl}/3\f$
  static SSR4 identity_vol(const TensorOptions & options = default_tensor_options());
  /// Create the deviatoric identity tensor \f$\delta_{ik}\delta_{jl}/2 + \delta_{il}\delta_{jk}/2 - \delta_{ij}\delta_{kl}/3\f$
  static SSR4 identity_dev(const TensorOptions & options = default_tensor_options());

  /// Building block for C1 constant
  static SSR4 identity_C1(const TensorOptions & options = default_tensor_options());
  /// Building block for C2 constant
  static SSR4 identity_C2(const TensorOptions & options = default_tensor_options());
  /// Building block for C3 constant
  static SSR4 identity_C3(const TensorOptions & options = default_tensor_options());

  /// Create the fourth order elasticity tensor given the Young's modulus and the Poisson's ratio.
  static SSR4 isotropic_E_nu(const Scalar & E, const Scalar & nu);
  /// Create the fourth order elasticity tensor given the Young's modulus and the Poisson's ratio.
  static SSR4 isotropic_E_nu(const double & E,
                             const double & nu,
                             const TensorOptions & options = default_tensor_options());

  /// Create the fourth order elasticity tensor given the three non-zero coefficients
  static SSR4 fill_C1_C2_C3(const Scalar & C1, const Scalar & C2, const Scalar & C3);
  /// Create the fourth order elasticity tensor given the three non-zero coefficients
  static SSR4 fill_C1_C2_C3(const double & C1,
                            const double & C2,
                            const double & C3,
                            const TensorOptions & options = default_tensor_options());

  /// Rotate
  SSR4 rotate(const Rot & r) const;

  /// Derivative of the rotated tensor w.r.t. the Rodrigues vector
  SSFR5 drotate(const Rot & r) const;

  /// Derivative of the rotated tensor w.r.t. itself
  SSSSR8 drotate_self(const Rot & r) const;

  /// Accessor
  Scalar operator()(Size i, Size j, Size k, Size l) const;

  /// Transpose minor axes, no-op
  SSR4 transpose_minor() const;

  /// Transpose major axes
  SSR4 transpose_major() const;
};

SSR4 operator*(const SSR4 & a, const SSR4 & b);
SR2 operator*(const SSR4 & a, const SR2 & b);
} // namespace neml2
