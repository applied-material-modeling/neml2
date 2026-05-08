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

#include "neml2/models/solid_mechanics/traction_separation/TractionSeparation.h"

namespace neml2
{
class Scalar;
class Vec;

/**
 * @brief Bilinear mixed-mode cohesive-zone traction-separation law with isotropic scalar damage.
 *
 * Implements a Camanho/Davila-style bilinear cohesive law with smooth Macaulay regularization on
 * the normal jump. The mode mixity \f$ \beta = \delta_s/\delta_n \f$, the damage-initiation jump
 * \f$ \delta_{init} \f$, and the full-degradation jump \f$ \delta_{final} \f$ are computed from
 * the previous-step displacement jump (lag_mode_mixity=true convention) so that the in-step
 * Jacobian remains tractable. The mixed-mode propagation criterion is the BK form.
 *
 * The current-step effective scalar separation
 * \f$ \delta_m = \sqrt{\langle\delta_n\rangle^2 + \delta_{s1}^2 + \delta_{s2}^2} \f$
 * uses a smooth Macaulay bracket with regularizer \f$ \alpha \f$. Damage evolves by a bilinear
 * law in \f$ \delta_m \f$ with irreversibility \f$ d = \max(d_{trial}, d_{old}) \f$. The
 * traction is
 * \f[
 *   \boldsymbol{T} = (1-d)\,K\,\boldsymbol{\delta}_{active}
 *                  + K\,\boldsymbol{\delta}_{inactive},
 * \f]
 * where the active part includes the regularized positive normal jump and both shear components,
 * and the inactive part is the (compressive) negative normal jump that resists penetration.
 *
 * Derivatives are computed by torch automatic differentiation through the regularized model.
 */
class BiLinearMixedModeTractionSeparation : public TractionSeparation
{
public:
  static OptionSet expected_options();

  BiLinearMixedModeTractionSeparation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;
  void request_AD() override;

  /// Current damage state output
  Variable<Scalar> & _d;

  /// Previous-step damage state (for irreversibility)
  const Variable<Scalar> & _d_old;

  /// Previous-step displacement jump (used for lagged mode mixity)
  const Variable<Vec> & _delta_old;

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

  /// Smoothing parameter for the regularized Heaviside
  const Scalar & _alpha;
};
} // namespace neml2
