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
 * @brief Wiechert (generalized Maxwell) viscoelastic model with two Maxwell branches.
 *
 * The Wiechert (generalized Maxwell) model is an equilibrium spring (modulus \f$ E_\infty \f$)
 * connected in parallel with \f$ N \f$ Maxwell branches. This implementation uses \f$ N = 2 \f$ to
 * give a finite-parameter Prony-series approximation of relaxation behavior; chain additional
 * Maxwell branches via input-file composition for higher-order responses. The total stress and the
 * viscous strain rates are
 * \f{align*}{
 *   \boldsymbol{\sigma} &= \left( E_\infty + E_1 + E_2 \right) \boldsymbol{\varepsilon} -
 *     E_1 \boldsymbol{\varepsilon}_{v,1} - E_2 \boldsymbol{\varepsilon}_{v,2}, \\
 *   \dot{\boldsymbol{\varepsilon}}_{v,i} &= E_i (\boldsymbol{\varepsilon} -
 *     \boldsymbol{\varepsilon}_{v,i}) / \eta_i, \quad i = 1, 2.
 * \f}
 */
class WiechertViscoelasticity : public Model
{
public:
  static OptionSet expected_options();

  WiechertViscoelasticity(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Total strain
  const Variable<SR2> & _E;

  /// Viscous strain in the first Maxwell branch
  const Variable<SR2> & _Ev1;

  /// Viscous strain in the second Maxwell branch
  const Variable<SR2> & _Ev2;

  /// Total stress
  Variable<SR2> & _S;

  /// Rate of viscous strain in the first Maxwell branch
  Variable<SR2> & _Ev1_dot;

  /// Rate of viscous strain in the second Maxwell branch
  Variable<SR2> & _Ev2_dot;

  /// Equilibrium spring modulus
  const Scalar & _Einf;

  /// First Maxwell branch modulus
  const Scalar & _E1;

  /// First Maxwell branch viscosity
  const Scalar & _eta1;

  /// Second Maxwell branch modulus
  const Scalar & _E2;

  /// Second Maxwell branch viscosity
  const Scalar & _eta2;
};
} // namespace neml2
