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
 * @brief Exponential damage traction-separation law with optional irreversible damage.
 *
 */
class ExpTractionSeparation : public TractionSeparationModel
{
public:
  static OptionSet expected_options();

  ExpTractionSeparation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Critical fracture energy
  const Scalar & _Gc;

  /// Softening length scale
  const Scalar & _delta0;

  /// Tangential weighting factor
  const Scalar & _beta;

  /// Regularization for effective displacement jump denominator
  double _eps;

  /// Whether to enforce damage irreversibility
  bool _irreversible_damage;

  /// Maximum effective displacement jump from the previous time step (old state input)
  const Variable<Scalar> & _delta_eff_max_old;

  /// Maximum effective displacement jump (state output)
  Variable<Scalar> & _delta_eff_max;
};
} // namespace neml2
