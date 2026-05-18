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
 * @brief Power-law (Alfano-Crisfield) mixed-mode propagation criterion: full (failure)
 * separation.
 *
 * Opening branch (\f$ \delta_n^+ > 0 \f$):
 * \f[
 *   \delta_f = \frac{2(1+\beta^2)}{K \delta_c}
 *     \left[ \left(\frac{1}{G_{Ic}}\right)^\eta + \left(\frac{\beta^2}{G_{IIc}}\right)^\eta
 *     \right]^{-1/\eta}.
 * \f]
 * Compression branch: \f$ \delta_f = 2 G_{IIc}/S \f$ (pure-shear closed form).
 *
 * `critical_separation` is declared as a nonlinear-capable parameter so the user can supply it
 * either as a plain numeric literal or wire it to an upstream
 * `CamanhoDavilaCriticalSeparation`.
 */
class PowerLawFullSeparation : public Model
{
public:
  static OptionSet expected_options();

  PowerLawFullSeparation(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  Variable<Scalar> & _to;
  const Variable<Scalar> & _dn;

  /// Mode-mixity ratio \f$ \beta \f$ (nonlinear-capable parameter)
  const Scalar & _beta;

  /// Critical (damage-onset) separation \f$ \delta_c \f$ (nonlinear-capable parameter)
  const Scalar & _delta_init;

  const Scalar & _K;
  const Scalar & _GIc;
  const Scalar & _GIIc;
  const Scalar & _S;
  const Scalar & _eta;
};
} // namespace neml2
