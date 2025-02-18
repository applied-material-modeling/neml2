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

#include "neml2/models/solid_mechanics/crystal_plasticity/SingleSlipHardeningRule.h"

namespace neml2
{
class Scalar;

/// Voce slip hardening when all slip systems share the same hardening value, \f$\dot{\bar{\tau}} = \theta_0 \left(1 - \frac{\bar{\tau}}{\tau_{sat}} \right) \sum \left|\dot{\gamma}_i \right|\f$
class VoceSingleSlipHardeningRule : public SingleSlipHardeningRule
{
public:
  static OptionSet expected_options();

  VoceSingleSlipHardeningRule(const OptionSet & options);

protected:
  /// Set the slip hardening rate
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Initial hardening slope
  const Scalar & _theta_0;

  /// Saturated hardening
  const Scalar & _tau_f;
};
} // namespace neml2
