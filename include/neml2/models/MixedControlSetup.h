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
class SR2;
class SSR4;

class MixedControlSetup : public Model
{
public:
  static OptionSet expected_options();

  MixedControlSetup(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  /// The threshold for the control signal
  const SR2 & _threshold;

  /// Actual input control signal
  const Variable<SR2> & _control;

  /// Driving forces to pull the "controlled" entries from
  const Variable<SR2> & _fixed_values;
  /// Conjugate state containing the mixed, unknown variables
  const Variable<SR2> & _mixed_state;

  /// Variable to target when the control signal is above the threshold
  Variable<SR2> & _above_var;
  /// Variable to target when the control signal is below the threshold
  Variable<SR2> & _below_var;

private:
  /// Construct the derivative operators from the control signal
  std::pair<SSR4, SSR4> make_operators(const SR2 & control) const;
};
} // namespace neml2
