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

/**
 * @brief Exponential cohesive-zone traction-separation law with isotropic scalar damage.
 *
 * Defines an effective scalar separation
 * \f[
 *   \delta_{\text{eff}} =
 *     \sqrt{\delta_n^2 + \beta\,(\delta_{s1}^2 + \delta_{s2}^2) + \varepsilon},
 * \f]
 * an exponential damage variable \f$ d = 1 - \exp(-\kappa/\delta_0) \f$, and a degraded linear
 * traction
 * \f[
 *   \boldsymbol{T} = (1 - d)\,\frac{G_c}{\delta_0^2}\,\boldsymbol{\delta},
 * \f]
 * where \f$ \kappa \f$ is either the current effective separation (reversible mode) or the
 * monotonically increasing maximum effective separation history
 * \f$ \kappa = \max(\kappa_n, \delta_{\text{eff}}) \f$ (irreversible mode). The small dtype-aware
 * regularizer \f$ \varepsilon \f$ — taken from `neml2::machine_precision()` — avoids the AD
 * singularity of \f$ \sqrt{\cdot} \f$ at zero jump.
 *
 * The model exposes \f$ \kappa \f$ as an internal state variable so that it can be advanced from
 * step to step through a transient driver. In reversible mode \f$ \kappa \f$ tracks the current
 * \f$ \delta_{\text{eff}} \f$ (no history is consulted); in irreversible mode it freezes at the
 * old maximum until the current effective separation exceeds it.
 */
class ExponentialTraction : public TractionSeparationLaw
{
public:
  static OptionSet expected_options();

  ExponentialTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Updated maximum effective separation (current state)
  Variable<Scalar> & _kappa;

  /// Old maximum effective separation (state from previous time step)
  const Variable<Scalar> & _kappa_n;

  /// Fracture toughness (critical energy release rate)
  const Scalar & _Gc;

  /// Softening length scale
  const Scalar & _delta0;

  /// Tangential weighting factor in the effective separation
  const Scalar & _beta;

  /// If true, freeze damage at the historical peak (no healing under unloading)
  const bool _irreversible;
};
} // namespace neml2
