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

#include "neml2/models/solid_mechanics/traction_separation/TractionSeparation.h"

namespace neml2
{
class Scalar;

/**
 * @brief 3D exponential cohesive-zone traction-separation law of Salehani and Irani.
 *
 * Defines a smooth exponential coupling of normal and shear separations:
 * \f[
 *   T_i = a_i \frac{\delta_i}{\delta_{u0,i}} \exp(-x), \qquad
 *   x = \frac{\delta_n}{\delta_{u0,n}}
 *       + \left(\frac{\delta_{s1}}{\delta_{u0,t}}\right)^2
 *       + \left(\frac{\delta_{s2}}{\delta_{u0,t}}\right)^2,
 * \f]
 * with \f$ a_n = e\, T_n^{\max} \f$ and \f$ a_t = \sqrt{2e}\, T_t^{\max} \f$. The normal component
 * enters linearly (\f$\alpha=1\f$) while the shear components enter quadratically (\f$\alpha=2\f$).
 *
 * Following the original implementation, the *internal* tangential characteristic length is
 * \f$ \sqrt{2}\, \delta_{u0,t} \f$ even though the user supplies \f$ \delta_{u0,t} \f$ directly.
 * Parameter values ported from references that use the same convention will reproduce the
 * published results exactly.
 *
 * No compression branch: the normal traction is negative for \f$ \delta_n < 0 \f$
 * (i.e. interpenetration produces an attractive force). Pair with a contact penalty if interface
 * compression is possible.
 */
class SalehaniIrani3DCTraction : public TractionSeparation
{
public:
  static OptionSet expected_options();

  SalehaniIrani3DCTraction(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Normal characteristic length (raw user input)
  const Scalar & _delta_u0_n;

  /// Tangential characteristic length (raw user input; sqrt(2) is applied internally)
  const Scalar & _delta_u0_t;

  /// Maximum normal traction
  const Scalar & _Tmax_n;

  /// Maximum shear traction
  const Scalar & _Tmax_t;
};
} // namespace neml2
