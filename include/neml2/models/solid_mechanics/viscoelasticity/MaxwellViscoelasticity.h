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
 * @brief Dashpot evolution equation for a Maxwell viscoelastic element.
 *
 * The Maxwell element consists of a linear spring and a Newtonian dashpot connected in series.
 * Total strain decomposes as \f$ \boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_e +
 * \boldsymbol{\varepsilon}_v \f$, the spring carries the entire stress \f$ \boldsymbol{\sigma} = E
 * \boldsymbol{\varepsilon}_e \f$, and the dashpot's viscous flow is governed by
 * \f$ \dot{\boldsymbol{\varepsilon}}_v = \boldsymbol{\sigma}/\eta \f$.
 * This object implements only the dashpot rate equation; it is intended to be composed with a
 * linear elasticity model to form a complete Maxwell element.
 */
class MaxwellViscoelasticity : public Model
{
public:
  static OptionSet expected_options();

  MaxwellViscoelasticity(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Stress acting across the dashpot
  const Variable<SR2> & _S;

  /// Rate of viscous strain
  Variable<SR2> & _Ev_dot;

  /// Dashpot viscosity
  const Scalar & _eta;
};
} // namespace neml2
