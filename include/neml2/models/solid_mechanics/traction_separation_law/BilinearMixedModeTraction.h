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

#include "neml2/models/solid_mechanics/traction_separation_law/TractionSeparationLaw.h"

namespace neml2
{
class Scalar;
class Vec;

/**
 * @brief Abstract base for bilinear mixed-mode cohesive-zone traction-separation laws.
 *
 * Camanho-Davila bilinear envelope in the mode-mixity plane combined with a mixed-mode
 * propagation criterion supplied by a concrete subclass. Damage is irreversible. The normal
 * compressive branch is restored through a Macaulay split so interpenetration does not soften
 * the interface. This base class owns all forward and Jacobian work *except* for the
 * criterion-specific computation of the full-degradation displacement jump
 * \f$ \delta_\text{final} \f$ in the opening branch, which is delegated to a virtual hook
 * implemented by each concrete subclass.
 */
class BilinearMixedModeTraction : public TractionSeparationLaw
{
public:
  static OptionSet expected_options();

  BilinearMixedModeTraction(const OptionSet & options);

protected:
  /// Inputs to the criterion-specific `delta_final` hook. Bundled so the signature stays
  /// stable as future criteria add or drop dependencies.
  struct DeltaFinalContext
  {
    /// Dtype-aware regularizer added inside `pow()` bases (and any other potentially singular
    /// operations) to keep derivatives bounded at zero displacement jump.
    double eps;
    /// Mode mixity \f$ \beta = \delta_s / \delta_n \f$, opening branch.
    const Scalar & beta;
    /// \f$ \beta^2 \f$, precomputed.
    const Scalar & beta_sq;
    /// Mixed-mode initiation jump \f$ \delta_\text{init} \f$, opening branch.
    const Scalar & delta_init_mixed;
    /// \f$ K \cdot \delta_\text{init} \f$, precomputed (appears in every criterion).
    const Scalar & Kdelta_init_mixed;
    /// \f$ \partial \beta / \partial \boldsymbol{\delta} \f$ in the opening branch.
    const Vec & dbeta_ddelta_open;
    /// \f$ \partial \delta_\text{init} / \partial \boldsymbol{\delta} \f$ in the opening branch.
    const Vec & ddelta_init_ddelta_open;
  };

  /// Outputs of the criterion-specific `delta_final` hook.
  struct DeltaFinalResult
  {
    /// \f$ \delta_\text{final} \f$ in the opening branch.
    Scalar value;
    /// \f$ \partial \delta_\text{final} / \partial \boldsymbol{\delta} \f$ in the opening
    /// branch. Only valid when the hook was invoked with `dout_din = true`.
    Vec ddelta_open;
  };

  /// Subclass hook: compute the criterion-specific \f$ \delta_\text{final} \f$ and (when
  /// requested) its Jacobian w.r.t. the current displacement jump.
  virtual DeltaFinalResult compute_delta_final(const DeltaFinalContext & ctx,
                                               bool dout_din) const = 0;

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

  /// Mixed-mode criterion exponent (interpretation is criterion-specific)
  const Scalar & _eta;
};
} // namespace neml2
