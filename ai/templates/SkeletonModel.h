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

// Replace with the actual parent header (e.g., "neml2/models/Model.h" or a domain base)
#include "neml2/models/Model.h"
// Include tensor types used by this model (Scalar, Vec, SR2, R2, ...)
#include "neml2/tensors/Scalar.h"

namespace neml2
{
/**
 * @brief One-line description of what this model computes.
 *
 * Optional: add one sentence on the governing equation if non-obvious.
 */
class SkeletonModel : public Model // Replace Model with actual base class
{
public:
  static OptionSet expected_options();

  SkeletonModel(const OptionSet & options);

protected:
  void set_value(bool out, bool dout_din, bool d2out_din2) override;

  // --- Input variables (declare_input_variable in constructor) ---
  /// Brief description of what this input represents
  const Variable<Scalar> & _input_var;

  // --- Output variables (declare_output_variable in constructor) ---
  /// Brief description of what this output represents
  Variable<Scalar> & _output_var;

  // --- Parameters (declare_parameter in constructor) ---
  /// Brief description of the parameter and its units
  const Scalar & _param;
};
} // namespace neml2
