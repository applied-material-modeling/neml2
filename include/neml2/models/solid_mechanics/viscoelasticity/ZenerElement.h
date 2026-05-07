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
 * @brief Zener (Standard Linear Solid) viscoelastic model.
 *
 * The Zener model is an equilibrium spring (modulus \f$ E_\infty \f$) connected in parallel with a
 * Maxwell branch (spring \f$ E_M \f$ and dashpot \f$ \eta_M \f$ in series). Letting \f$
 * \boldsymbol{\varepsilon}_v \f$ be the viscous strain in the Maxwell branch's dashpot, the Maxwell
 * branch stress is \f$ \boldsymbol{\sigma}_M = E_M (\boldsymbol{\varepsilon} -
 * \boldsymbol{\varepsilon}_v) \f$ and the total stress reduces to \f$ \boldsymbol{\sigma} =
 * (E_\infty + E_M) \boldsymbol{\varepsilon} - E_M \boldsymbol{\varepsilon}_v \f$. The viscous
 * strain evolves according to \f$ \dot{\boldsymbol{\varepsilon}}_v = \boldsymbol{\sigma}_M / \eta_M
 * \f$.
 */
class ZenerElement : public Model
{
public:
  static OptionSet expected_options();

  ZenerElement(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Total strain
  const Variable<SR2> & _E;

  /// Viscous strain in the Maxwell branch
  const Variable<SR2> & _Ev;

  /// Total stress
  Variable<SR2> & _S;

  /// Rate of viscous strain
  Variable<SR2> & _Ev_dot;

  /// Equilibrium spring modulus
  const Scalar & _Einf;

  /// Maxwell branch modulus
  const Scalar & _EM;

  /// Maxwell branch viscosity
  const Scalar & _etaM;
};
} // namespace neml2
