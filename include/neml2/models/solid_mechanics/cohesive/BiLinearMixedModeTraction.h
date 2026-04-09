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

#include "neml2/models/solid_mechanics/cohesive/TractionSeparationModel.h"

namespace neml2
{
class Scalar;

/**
 * @brief Bilinear mixed-mode cohesive zone model with irreversible damage (Camanho & Davila 2002).
 *
 */
class BiLinearMixedModeTraction : public TractionSeparationModel
{
public:
  static OptionSet expected_options();

  BiLinearMixedModeTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  ///@{
  /// Model parameters
  const Scalar & _K;       ///< Penalty elastic stiffness
  const Scalar & _GI_c;   ///< Mode I critical energy release rate
  const Scalar & _GII_c;  ///< Mode II critical energy release rate
  const Scalar & _N;       ///< Tensile (normal) strength
  const Scalar & _S;       ///< Shear strength
  const Scalar & _eta;     ///< Mixed-mode propagation exponent
  const Scalar & _viscosity; ///< Viscous regularization coefficient (0 = none)
  ///@}

  ///@{
  /// Additional input/output variables
  /// Damage variable from the previous time step
  const Variable<Scalar> & _d_old;
  /// Old displacement jump (for lagged mode-mixity computation)
  const Variable<Vec> & _delta_old;
  /// Scalar time at start of step (OLD_FORCES) for dt computation
  const Variable<Scalar> & _t_old;
  /// Scalar time at end of step (FORCES) for dt computation
  const Variable<Scalar> & _t;
  /// Updated damage variable (output state)
  Variable<Scalar> & _d;
  ///@}

  ///@{
  /// Control flags read from options at construction
  bool _lag_mode_mixity;   ///< Use lagged delta for mode-mixity ratio
  bool _lag_disp_jump;     ///< Use lagged delta for effective displacement jump
  std::string _criterion;  ///< "BK" or "POWER_LAW"
  ///@}
};
} // namespace neml2
