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
class Vec;

/**
 * @brief Common base for cohesive-zone traction-separation laws.
 *
 * A traction-separation law (TSL) maps a 3-vector displacement jump in the local interface frame
 * \f$ \boldsymbol{\delta} = [\delta_n, \delta_{s1}, \delta_{s2}] \f$ to a 3-vector traction
 * \f$ \boldsymbol{T} = [T_n, T_{s1}, T_{s2}] \f$ in the same frame. Concrete subclasses provide the
 * specific constitutive relation \f$ \boldsymbol{T}(\boldsymbol{\delta}) \f$ and may declare
 * additional parameters, internal state variables (e.g. damage), and option flags.
 */
class TractionSeparationLaw : public Model
{
public:
  static OptionSet expected_options();

  TractionSeparationLaw(const OptionSet & options);

protected:
  /// Displacement jump in the local interface frame [delta_n, delta_s1, delta_s2]
  const Variable<Vec> & _delta;

  /// Traction in the local interface frame [T_n, T_s1, T_s2]
  Variable<Vec> & _T;
};
} // namespace neml2
