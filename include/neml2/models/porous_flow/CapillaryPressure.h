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
/// Base class for capillary pressure models in porous flow.
class CapillaryPressure : public Model
{
public:
  static OptionSet expected_options();

  CapillaryPressure(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_dS, bool d2out_dS2) override;

  /// Calculate the capillary pressure without logarithmic extension.
  virtual std::tuple<Scalar, Scalar, Scalar>
  calculate_pressure(const Scalar & S, bool out, bool dout_dS, bool d2out_dS2) const = 0;

  /// Effective saturation
  const Variable<Scalar> & _S;

  /// Capillary pressure
  Variable<Scalar> & _Pc;

  /// Whether to apply logarithmic extension
  const bool _log_extension;

  /// Whether the transition saturation is specified
  const bool _Sp_specified;

  /// Transistion saturation
  const double _Sp;
};
}
