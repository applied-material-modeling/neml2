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

#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

/**
 * @brief Benzeggagh-Kenane (BK) mixed-mode propagation criterion: full-degradation jump.
 *
 * Opening branch (\f$ \delta_n^+ > 0 \f$):
 * \f[
 *   \delta_\text{final} = \frac{2}{K \delta_\text{init}}
 *     \left( G_{Ic} + (G_{IIc} - G_{Ic}) \left(\frac{\beta^2}{1+\beta^2}\right)^\eta \right).
 * \f]
 * Compression branch: \f$ \delta_\text{final} = 2 G_{IIc}/S \f$ (pure-shear closed form).
 */
class BKCriterion : public Model
{
public:
  static OptionSet expected_options();

  BKCriterion(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Full-degradation displacement jump \f$ \delta_\text{final} \f$
  Variable<Scalar> & _to;

  /// Mode-mixity ratio \f$ \beta \f$
  const Variable<Scalar> & _beta;

  /// Initiation displacement jump \f$ \delta_\text{init} \f$
  const Variable<Scalar> & _delta_init;

  /// Macaulay-positive normal jump (only needed to determine the opening/compression branch)
  const Variable<Scalar> & _dn_pos;

  const Scalar & _K;
  const Scalar & _GIc;
  const Scalar & _GIIc;
  const Scalar & _S;
  const Scalar & _eta;
};
} // namespace neml2
