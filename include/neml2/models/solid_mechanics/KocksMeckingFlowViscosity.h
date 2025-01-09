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

#include "neml2/models/NonlinearParameter.h"

namespace neml2
{
class Scalar;

/**
 * @brief A scalar-valued parameter defined by (mu0 - D/(exp(T0/T)-1)) * exp(_C)
 */
class KocksMeckingFlowViscosity : public NonlinearParameter<Scalar>
{
public:
  static OptionSet expected_options();

  KocksMeckingFlowViscosity(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The Kocks-Mecking slope
  const Scalar & _A;

  /// The Kocks-Mecking intercept
  const Scalar & _B;

  /// The shear modulus
  const Scalar & _mu;

  /// The reference strain rate
  const Real _eps0;

  /// The Boltzmann constant
  const Real _k;

  /// Burgers vector cubed
  const Real _b3;

  /// The temperature
  const Variable<Scalar> & _T;
};
}
