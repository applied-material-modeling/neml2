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

#include "neml2/base/EnumSelection.h"
#include "neml2/models/solid_mechanics/traction_separation/TractionSeparation.h"

namespace neml2
{
class Scalar;

/**
 * @brief Bilinear mixed-mode cohesive-zone traction-separation law with isotropic scalar damage.
 *
 * Camanho-Davila bilinear envelope in the mode-mixity plane combined with the Benzeggagh-Kenane
 * (BK) or power-law mixed-mode propagation criterion. Damage is irreversible. The normal
 * compressive branch is restored through a Macaulay split so interpenetration does not soften
 * the interface. Variable Jacobians \f$ \partial T/\partial \delta \f$, \f$ \partial T/\partial
 * d_{old} \f$, \f$ \partial d/\partial \delta \f$, and \f$ \partial d/\partial d_{old} \f$ are
 * derived analytically.
 */
class BiLinearMixedModeTractionSeparation : public TractionSeparation
{
public:
  static OptionSet expected_options();

  BiLinearMixedModeTractionSeparation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Current damage state output
  Variable<Scalar> & _d;

  /// Previous-step damage state (for irreversibility)
  const Variable<Scalar> & _d_old;

  /// Penalty elastic stiffness (same in normal and tangential directions)
  const Scalar & _K;

  /// Mode I critical energy release rate
  const Scalar & _GIc;

  /// Mode II critical energy release rate
  const Scalar & _GIIc;

  /// Tensile (normal) strength
  const Scalar & _N;

  /// Shear strength
  const Scalar & _S;

  /// Power-law exponent in the BK / POWER_LAW criterion
  const Scalar & _eta;

  /// Mixed-mode propagation criterion: BK or POWER_LAW
  const EnumSelection _criterion;

  /// Small regularizer added inside sqrt() to keep its derivative bounded at zero jump
  const double _eps;
};
} // namespace neml2
