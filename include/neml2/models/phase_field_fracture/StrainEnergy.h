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
#include "neml2/tensors/SR2.h"

namespace neml2
{
class StrainEnergy : public Model
{
public:
  static OptionSet expected_options();

  StrainEnergy(const OptionSet & options);

protected:
  /**
   * Whether this model describes compliance. When set to true, we compute stress (rate) from strain
   * (rate). When set to false, we compute strain (rate) from stress (rate).
   */
  // const bool _compliance;

  /// Whether this model is in rate form. If true, a "_rate" suffix is appended to the variables.
  // const bool _rate_form;

  /// The strain (rate) variable accessor
  const VariableName _strain;

  /// The stress (rate) variable accessor
  // const VariableName _stress;

  /**
   * The variable accessor for the input. If _compliance == true, this is the stress (rate).
   * Otherwise this is the strain (rate).
   */
  const Variable<SR2> & _from;

  /**
   * The variable accessor for the output. If _compliance == true, this is the strain (rate).
   * Otherwise this is the stress (rate).
   */
  // Variable<SR2> & _to;
  /// elastic strain energy density
  Variable<Scalar> & _psie;
};
} // namespace neml2
