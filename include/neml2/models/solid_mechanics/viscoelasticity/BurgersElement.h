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
class SR2;

/**
 * @brief Burgers viscoelastic model (Maxwell element in series with a Kelvin-Voigt element).
 *
 * The Burgers model is a Maxwell element (spring \f$ E_M \f$ and dashpot \f$ \eta_M \f$ in series)
 * connected in series with a Kelvin-Voigt element (spring \f$ E_K \f$ and dashpot \f$ \eta_K \f$ in
 * parallel). The total strain decomposes as
 * \f$ \boldsymbol{\varepsilon} = \boldsymbol{\sigma}/E_M + \boldsymbol{\varepsilon}_{v,M} +
 * \boldsymbol{\varepsilon}_{K} \f$, where \f$ \boldsymbol{\varepsilon}_{v,M} \f$ is the Maxwell
 * dashpot's viscous strain and \f$ \boldsymbol{\varepsilon}_{K} \f$ is the Kelvin-Voigt strain.
 * The shared series stress is then \f$ \boldsymbol{\sigma} = E_M (\boldsymbol{\varepsilon} -
 * \boldsymbol{\varepsilon}_{v,M} - \boldsymbol{\varepsilon}_{K}) \f$, and the internal-strain
 * evolution equations are
 * \f{align*}{
 *   \dot{\boldsymbol{\varepsilon}}_{v,M} &= \boldsymbol{\sigma} / \eta_M, \\
 *   \dot{\boldsymbol{\varepsilon}}_{K} &= (\boldsymbol{\sigma} - E_K
 *     \boldsymbol{\varepsilon}_{K}) / \eta_K.
 * \f}
 */
class BurgersElement : public Model
{
public:
  static OptionSet expected_options();

  BurgersElement(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Total strain
  const Variable<SR2> & _E;

  /// Maxwell dashpot viscous strain
  const Variable<SR2> & _EvM;

  /// Kelvin-Voigt branch strain
  const Variable<SR2> & _EK;

  /// Total stress
  Variable<SR2> & _S;

  /// Rate of Maxwell dashpot viscous strain
  Variable<SR2> & _EvM_dot;

  /// Rate of Kelvin-Voigt branch strain
  Variable<SR2> & _EK_dot;

  /// Maxwell branch spring modulus
  const Scalar & _EM;

  /// Maxwell branch dashpot viscosity
  const Scalar & _etaM;

  /// Kelvin-Voigt branch spring modulus
  const Scalar & _EK_param;

  /// Kelvin-Voigt branch dashpot viscosity
  const Scalar & _etaK;
};
} // namespace neml2
