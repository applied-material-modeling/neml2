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

/// Standard forest hardening model, per slip system dislocation density
//  \f$\dot{\rho}_i =  \left( k_1 \sqrt{\rho_i} - k_2 \rho_i \right) \left| \dot{\gamma}_i \right|
//  \f$
class PerSlipForestDislocationEvolution : public Model
{
public:
  static OptionSet expected_options();

  PerSlipForestDislocationEvolution(const OptionSet & options);

protected:
  /// Set the slip hardening rate
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// Forest hardening rate
  Variable<Scalar> & _rho_dot;

  /// Current dislocation density
  const Variable<Scalar> & _rho;

  /// Slip rates
  const Variable<Scalar> & _gamma_dot;

  /// Hardening coefficient
  const Scalar & _k1;

  /// Recovery coefficient
  const Scalar & _k2;
};
} // namespace neml2
