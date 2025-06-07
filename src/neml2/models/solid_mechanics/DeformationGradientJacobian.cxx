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

#include "neml2/models/solid_mechanics/DeformationGradientJacobian.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"

namespace neml2
{
register_NEML2_object(DeformationGradientJacobian);

OptionSet
DeformationGradientJacobian::expected_options()
{
  auto options = Model::expected_options();
  options.doc() = "Calculate the Jacobian of the deformation gradient tensor \\f$ F \\f$, aka \\f$ "
                  "J = det(F) \\f$.";

  options.set<VariableName>("deformation_gradient") = VariableName(STATE, "F");
  options.set("deformation_gradient").doc() = "Deformation gradient tensor";

  options.set<VariableName>("jacobian") = VariableName(STATE, "J");
  options.set("jacobian").doc() = "The jacobian";

  return options;
}

DeformationGradientJacobian::DeformationGradientJacobian(const OptionSet & options)
  : Model(options),
    _F(declare_input_variable<R2>("deformation_gradient")),
    _J(declare_output_variable<Scalar>("jacobian"))
{
}

void
DeformationGradientJacobian::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  if (out)
  {
    _J = R2(_F).det();
  }

  if (dout_din)
    if (_F.is_dependent())
    {
      _J.d(_F) = R2(_F).det() * R2(_F).inverse().transpose();
    }
}
} // namespace neml2
