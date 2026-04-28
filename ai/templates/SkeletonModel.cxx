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

#include "neml2/models/<domain>/SkeletonModel.h"
#include "neml2/tensors/Scalar.h"

namespace neml2
{
// This macro registers the class with the NEML2 factory. MUST be present.
register_NEML2_object(SkeletonModel);

OptionSet
SkeletonModel::expected_options()
{
  // Always call parent first
  OptionSet options = Model::expected_options(); // Replace with actual base class
  options.doc() = "One-line description for the input file parser / user docs.";

  // Enable this only if the model provides analytical second derivatives in set_value.
  // Omit entirely if second derivatives are not implemented.
  options.set_private<bool>("define_second_derivatives", true);

  // Declare a scalar parameter exposed under the input file key "my_parameter".
  options.add_parameter<Scalar>("my_parameter",
                                "Description of the parameter and its physical units");

  // Declare an input variable exposed under the input file key "input_name".
  options.add_input("input_name", "Description of the input variable");

  // Declare an output variable exposed under the input file key "output_name".
  options.add_output("output_name", "Description of the output variable");

  // Plain (non-parameter, non-variable) options use add<T>:
  // options.add<bool>("my_flag", "What this flag controls");
  // options.add<double>("my_const", "Description");

  return options;
}

SkeletonModel::SkeletonModel(const OptionSet & options)
  : Model(options), // Replace with actual base class
    _input_var(declare_input_variable<Scalar>("input_name")),
    _output_var(declare_output_variable<Scalar>("output_name")),
    // declare_parameter: short internal name "p" bound to option key "my_parameter".
    // Pass /*allow_nonlinear=*/true to expose the parameter as nonlinear (AD-tracked).
    _param(declare_parameter<Scalar>("p", "my_parameter"))
{
}

void
SkeletonModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // --- Forward value ---
  if (out)
  {
    // Use operator() to get the tensor value from a Variable<T>
    _output_var = _param * _input_var;
  }

  // --- First derivative (Jacobian) ---
  if (dout_din)
  {
    // Guard with is_dependent() to avoid computing derivatives of constants
    if (_input_var.is_dependent())
      _output_var.d(_input_var) = _param;

    // For parameter derivatives, use declare_parameter with AD tracking (optional):
    // _output_var.d(_param_var) = _input_var;
  }

  // --- Second derivative ---
  if (d2out_din2)
  {
    // zero for linear models — leave block empty, do NOT remove it
  }
}
} // namespace neml2
